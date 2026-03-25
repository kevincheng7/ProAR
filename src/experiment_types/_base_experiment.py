from __future__ import annotations

import inspect
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from tensordict import TensorDict
from torch import Tensor

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.dataset_dimensions import get_dims_of_dataset
from src.models._base_model import BaseModel
from src.utilities.evaluation import evaluate_ensemble_prediction
from src.utilities.utils import get_logger, rrearrange, torch_to_numpy


class BaseExperiment(LightningModule):
    r"""Inference-only base class for ML experiments.

    Methods that need to be implemented by your concrete ML model:
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.

    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model()).

    Args:
        model_config: DictConfig with the model configuration
        datamodule_config: DictConfig with the datamodule configuration
        diffusion_config: DictConfig with the diffusion configuration
        enable_inference_dropout (bool): Whether to enable dropout during inference. Default: False
        num_predictions (int): The number of predictions to make for each input sample
        prediction_inputs_noise (float): The amount of noise to add to the inputs before predicting
        verbose (bool): Whether to print/log or not
    """

    CHANNEL_DIM = -3

    def __init__(
        self,
        model_config: DictConfig,
        datamodule_config: DictConfig,
        diffusion_config: Optional[DictConfig] = None,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        enable_inference_dropout: bool = False,
        num_predictions: int = 1,
        logging_infix: str = "",
        prediction_inputs_noise: float = 0.0,
        seed: int = None,
        verbose: bool = True,
        name: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config", "datamodule_config", "diffusion_config", "verbose"])
        self.log_text = get_logger(name=self.__class__.__name__)
        self.verbose = verbose
        self.logging_infix = logging_infix
        if not self.verbose:
            self.log_text.setLevel(logging.WARN)

        self._datamodule = None
        self.model_config = model_config
        self.datamodule_config = datamodule_config
        self.diffusion_config = diffusion_config
        self.is_diffusion_model = diffusion_config is not None and diffusion_config.get("_target_", None) is not None
        self.dims = get_dims_of_dataset(self.datamodule_config)
        self.model = self.instantiate_model()

        self.model.ema_scope = self.ema_scope

        if enable_inference_dropout:
            self.log_text.info(" Enabling dropout during inference!")

        self._start_validation_epoch_time = self._start_test_epoch_time = None
        self._validation_step_outputs, self._predict_step_outputs = [], []
        self._test_step_outputs = defaultdict(list)

        self._val_metrics = self._test_metrics = self._predict_metrics = None

        self._check_args()

        if self.use_ensemble_predictions("val"):
            self.log_text.info(f" Using a {num_predictions}-member ensemble for validation.")

        if hasattr(self.model, "example_input_array"):
            self.example_input_array = self.model.example_input_array

    # --------------------------------- Interface with model
    def actual_num_input_channels(self, num_input_channels: int) -> int:
        return num_input_channels

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        return num_output_channels

    @property
    def num_conditional_channels(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs."""
        nc = self.dims.get("conditional", 0)
        if self.is_diffusion_model:
            d_class = self.diffusion_config.get("_target_").lower()
            is_standard_diffusion = "dyffusion" not in d_class
            if is_standard_diffusion:
                nc += self.window * self.dims["input"]

            fwd_cond = self.diffusion_config.get("forward_conditioning", "").lower()
            if fwd_cond == "":
                pass
            elif fwd_cond == "data+noise":
                nc += 2 * self.window * self.dims["input"]
            elif fwd_cond in ["none", None]:
                pass
            else:
                nc += self.window * self.dims["input"]
        return nc

    @property
    def window(self) -> int:
        return self.datamodule_config.get("window", 1)

    @property
    def horizon(self) -> int:
        return self.datamodule_config.horizon

    @property
    def datamodule(self) -> BaseDataModule:
        if self._datamodule is None:
            self._datamodule = self.trainer.datamodule
        return self._datamodule

    def instantiate_model(self, *args, **kwargs) -> BaseModel:
        r"""Instantiate the model."""
        in_channels = self.actual_num_input_channels(self.dims["input"])
        out_channels = self.actual_num_output_channels(self.dims["output"])
        cond_channels = self.num_conditional_channels
        kwargs["datamodule_config"] = self.datamodule_config

        model = hydra.utils.instantiate(
            self.model_config,
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            num_conditional_channels=cond_channels,
            spatial_shape=self.dims["spatial"],
            _recursive_=False,
            **kwargs,
        )
        self.log_text.info(
            f"Instantiated model: {model.__class__.__name__}, with"
            f" # input/output/conditional channels: {in_channels}, {out_channels}, {cond_channels}"
        )
        if self.is_diffusion_model:
            model = hydra.utils.instantiate(self.diffusion_config, model=model, _recursive_=False, **kwargs)
            self.log_text.info(
                f"Instantiated diffusion model: {model.__class__.__name__}, with"
                f" #diffusion steps={model.num_timesteps}"
            )
        return model

    def forward(self, *args, **kwargs) -> Any:
        y = self.model(*args, **kwargs)
        return y

    # --------------------------------- Names
    @property
    def short_description(self) -> str:
        return self.__class__.__name__

    @property
    def test_set_name(self) -> str:
        return self.trainer.datamodule.test_set_name if hasattr(self.trainer.datamodule, "test_set_name") else "test"

    @property
    def prediction_set_name(self) -> str:
        return (
            self.trainer.datamodule.prediction_set_name
            if hasattr(self.trainer.datamodule, "prediction_set_name")
            else "predict"
        )

    # --------------------------------- Metrics
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        raise NotImplementedError(
            f"Please implement the method 'get_metrics' in your concrete class {self.__class__.__name__}"
        )

    @property
    def default_monitor_metric(self) -> str:
        return "val/mse"

    @property
    def val_metrics(self):
        if self._val_metrics is None:
            self._val_metrics = self.get_metrics(split="val", split_name="val").to(self.device)
        return self._val_metrics

    @property
    def test_metrics(self):
        if self._test_metrics is None:
            self._test_metrics = self.get_metrics(split="test", split_name=self.test_set_name).to(self.device)
        return self._test_metrics

    @property
    def predict_metrics(self):
        if self._predict_metrics is None:
            self._predict_metrics = self.get_metrics(split="predict", split_name=self.prediction_set_name).to(
                self.device
            )
        return self._predict_metrics

    # --------------------------------- Check arguments for validity
    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    @contextmanager
    def ema_scope(self, context=None, force_non_ema: bool = False, condition: bool = None):
        """No-op context manager (EMA removed for inference-only mode)."""
        yield None

    @contextmanager
    def inference_dropout_scope(self, condition: bool = None, context=None):
        """Context manager to switch to inference dropout mode.
        Args:
            condition (bool, optional): If True, switch to inference dropout mode. If False, switch to training mode.
                If None, use the value of self.hparams.enable_inference_dropout.
            context (str, optional): If not None, print this string when switching to inference dropout mode.
        """
        condition = self.hparams.enable_inference_dropout if condition is None else condition
        if condition:
            self.model.enable_inference_dropout()
            if context is not None:
                self.log_text.debug(f"{context}: Switched to enabled inference dropout")
        try:
            yield None
        finally:
            if condition:
                self.model.disable_inference_dropout()
                if context is not None:
                    self.log_text.debug(f"{context}: Switched to disabled inference dropout")

    @contextmanager
    def timing_scope(self, context="", no_op=True, precision=2):
        """Context manager to measure the time of the code inside the context. (By default, does nothing.)"""
        start_time = time.time() if not no_op else None
        try:
            yield None
        finally:
            if not no_op:
                context = f"``{context}``:" if context else ""
                self.log_text.info(f"Elapsed time {context} {time.time() - start_time:.{precision}f}s")

    def predict(
        self,
        inputs: Union[Tensor, Dict],
        num_predictions: Optional[int] = None,
        reshape_ensemble_dim: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        This should be the main method to use for making predictions/doing inference.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`.
            num_predictions (int, optional): Number of predictions to make. If None, use the default value.
            reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions in a post-processed format.
        """
        base_num_predictions = self.hparams.num_predictions
        self.hparams.num_predictions = num_predictions or base_num_predictions
        if (
            hasattr(self.model, "sample_loop")
            and "num_predictions" in inspect.signature(self.model.sample_loop).parameters
        ):
            kwargs["num_predictions"] = self.hparams.num_predictions

        results = self.model.predict_forward(inputs, **kwargs)
        if torch.is_tensor(results):
            results = {"preds": results}

        self.hparams.num_predictions = base_num_predictions
        results = self.reshape_predictions(results, reshape_ensemble_dim)
        results = self.unpack_predictions(results)

        return results

    def reshape_predictions(
        self, results: Dict[str, Tensor], reshape_ensemble_dim: bool = True, add_ensemble_dim_in_inputs: bool = False
    ) -> Dict[str, Tensor]:
        """Reshape and unpack the predictions from the model. This modifies the input dictionary in-place."""
        ensemble_size = self.hparams.num_predictions
        pred_keys = [k for k in results.keys() if "preds" in k]
        preds_shape = results[pred_keys[0]].shape
        if reshape_ensemble_dim and preds_shape[0] > 1:
            if add_ensemble_dim_in_inputs or (ensemble_size > 1 and preds_shape[0] % ensemble_size == 0):
                results = self._reshape_ensemble_preds(results, "predict")
                preds_shape = results[pred_keys[0]].shape
        return results

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place."""
        if self._trainer is None:
            return results
        return results

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results)
        Returns:
            results_dict: Dict[str, Tensor], where for each semantically different result, a separate prefix key is used
                Then, for each prefix key <p>, results_dict must contain <p>_preds and <p>_targets.
        """
        raise NotImplementedError(f"Please implement the _evaluation_step method for {self.__class__.__name__}")

    def evaluation_step(self, batch: Any, batch_idx: int, split: str, **kwargs) -> Dict[str, Tensor]:
        if "boundary_conditions" in inspect.signature(self._evaluation_step).parameters.keys():
            kwargs["boundary_conditions"] = self.datamodule.boundary_conditions
            kwargs.update(self.datamodule.get_boundary_condition_kwargs(batch, batch_idx, split))

        with self.ema_scope():
            with self.inference_dropout_scope():
                return self._evaluation_step(batch, batch_idx, split, **kwargs)

    def use_ensemble_predictions(self, split: str) -> bool:
        return self.hparams.num_predictions > 1 and split in ["val", "test", "predict"]

    def use_stacked_ensemble_inputs(self, split: str) -> bool:
        return True

    def get_ensemble_inputs(
        self,
        inputs_raw: Optional[Union[Tensor, Dict]],
        split: str,
        add_noise: bool = True,
        flatten_into_batch_dim: bool = True,
    ) -> Optional[Union[Tensor, Dict]]:
        """Get the inputs for the ensemble predictions"""
        if inputs_raw is None:
            return None
        elif not self.use_stacked_ensemble_inputs(split):
            return inputs_raw
        elif self.use_ensemble_predictions(split):
            num_predictions = self.hparams.num_predictions
            if isinstance(inputs_raw, dict):
                inputs = {
                    k: self.get_ensemble_inputs(v, split, add_noise, flatten_into_batch_dim)
                    for k, v in inputs_raw.items()
                }
            else:
                if isinstance(inputs_raw, Sequence):
                    inputs = np.array([inputs_raw] * num_predictions)
                elif add_noise:
                    inputs = torch.stack(
                        [
                            inputs_raw + self.inputs_noise * torch.randn_like(inputs_raw)
                            for _ in range(num_predictions)
                        ],
                        dim=0,
                    )
                else:
                    inputs = torch.stack([inputs_raw for _ in range(num_predictions)], dim=0)

                if flatten_into_batch_dim:
                    inputs = rrearrange(inputs, "N B ... -> (N B) ...")
        else:
            inputs = inputs_raw
        return inputs

    def _reshape_ensemble_preds(self, results: Dict[str, Tensor], split: str) -> Dict[str, Tensor]:
        r"""
        Reshape the predictions of an ensemble to the correct shape,
        where the first dimension is the ensemble dimension, N.
        """
        num_predictions = self.hparams.num_predictions
        if self.use_ensemble_predictions(split):
            if not self.use_stacked_ensemble_inputs(split):
                return results

            for key in results:
                if "targets" not in key and "true" not in key:
                    b = results[key].shape[0]
                    assert (
                        b % num_predictions == 0
                    ), f"key={key}: b % #ens_mems = {b} % {num_predictions} != 0 ...Did you forget to create the input ensemble?"
                    batch_size = max(1, b // num_predictions)
                    results[key] = results[key].reshape(num_predictions, batch_size, *results[key].shape[1:])
        return results

    def _evaluation_get_preds(self, outputs: List[Any]) -> Dict[str, Union[torch.distributions.Normal, np.ndarray]]:
        num_predictions = self.hparams.num_predictions
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        outputs_keys, results = outputs[0].keys(), dict()
        for key in outputs_keys:
            first_output = outputs[0][key]
            is_tensor_dict = isinstance(first_output, TensorDict)
            is_normal_dict = isinstance(first_output, dict)
            if is_normal_dict:
                s1, s2 = first_output[list(first_output.keys())[0]].shape[:2]
            else:
                s1, s2 = first_output.shape[:2]
            batch_axis = 1 if (s1 == num_predictions and "targets" not in key and "true" not in key) else 0
            if is_normal_dict:
                results[key] = {
                    k: np.concatenate([out[key][k] for out in outputs], axis=batch_axis) for k in first_output.keys()
                }
            elif is_tensor_dict:
                results[key] = torch.cat([out[key] for out in outputs], dim=batch_axis)
            else:
                try:
                    results[key] = np.concatenate([out[key] for out in outputs], axis=batch_axis)
                except ValueError as e:
                    raise ValueError(
                        f"Error when concatenating {key}: {e}.\n"
                        f"Shape 0: {first_output.shape}, -1: {outputs[-1][key].shape}"
                    )

        return results

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        results = self.evaluation_step(batch, batch_idx, split="val", dataloader_idx=dataloader_idx, **kwargs)
        results = torch_to_numpy(results)
        self._validation_step_outputs.append(results)
        return results

    def ensemble_logging_infix(self, split: str) -> str:
        """No '/' in front of the infix! But '/' at the end!"""
        s = "" if self.logging_infix == "" else f"{self.logging_infix}/".replace("//", "/")
        if self.inputs_noise > 0.0 and split != "val":
            s += f"{self.inputs_noise}eps/"
        s += f"{self.hparams.num_predictions}ens_mems/"
        return s

    def _eval_ensemble_predictions(self, outputs: List[Any], split: str):
        if not self.use_ensemble_predictions(split):
            return
        numpy_results = self._evaluation_get_preds(outputs)

        all_preds_metrics = defaultdict(list)
        preds_keys = [k for k in numpy_results.keys() if k.endswith("preds")]
        for preds_key in preds_keys:
            prefix = preds_key.split("_")[0] if preds_key != "preds" else ""
            targets_key = f"{prefix}_targets" if prefix else "targets"
            metrics = evaluate_ensemble_prediction(numpy_results[preds_key], targets=numpy_results[targets_key])
            preds_key_metrics = dict()
            for m, v in metrics.items():
                preds_key_metrics[f"{split}/{self.ensemble_logging_infix(split)}{prefix}/{m}"] = v
                all_preds_metrics[f"{split}/{self.ensemble_logging_infix(split)}avg/{m}"].append(v)

            self.log_dict(preds_key_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        avg_metrics = {k: np.mean(v) for k, v in all_preds_metrics.items()}
        self.log_dict(avg_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._eval_ensemble_predictions(self._validation_step_outputs, split="val")
        self._validation_step_outputs = []
        val_time = time.time() - self._start_validation_epoch_time
        val_stats = {
            "time/validation": val_time,
            "num_predictions": self.hparams.num_predictions,
            "noise_level": self.inputs_noise,
            "epoch": float(self.current_epoch),
            "global_step": self.global_step,
        }
        self.log_dict({**val_stats, "epoch": float(self.current_epoch)}, prog_bar=False)
        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        split = "test" if dataloader_idx is None else self.test_set_names[dataloader_idx]
        results = self.evaluation_step(batch, batch_idx, split="test", **kwargs)
        results = torch_to_numpy(results)
        self._test_step_outputs[split].append(results)
        return results

    def on_test_epoch_end(self, calc_ensemble_metrics: bool = True):
        for test_split in self._test_step_outputs.keys():
            if calc_ensemble_metrics:
                self._eval_ensemble_predictions(self._test_step_outputs[test_split], split=test_split)
        self._test_step_outputs = defaultdict(list)
        test_time = time.time() - self._start_test_epoch_time
        self.log_dict({"time/test": test_time}, prog_bar=False, sync_dist=True)

    # ---------------------------------------------------------------------- Inference
    def on_predict_start(self) -> None:
        for pdl in self.trainer.predict_dataloaders:
            assert pdl.dataset.dataset_id == "predict", f"dataset_id is not 'predict', but {pdl.dataset.dataset_id}"

        n_preds = self.hparams.num_predictions
        if n_preds > 1:
            self.log_text.info(f"Generating {n_preds} predictions per input with noise level {self.inputs_noise}")
        if "autoregressive_steps" in self.hparams:
            self.log_text.info(f"Autoregressive steps: {self.hparams.autoregressive_steps}")

    @property
    def inputs_noise(self):
        return self.hparams.prediction_inputs_noise

    def on_predict_epoch_start(self) -> None:
        if self.inputs_noise > 0:
            self.log_text.info(f"Adding noise to inputs with level {self.inputs_noise}")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        results = self.evaluation_step(batch, batch_idx, split="predict", **kwargs)
        results = torch_to_numpy(results)
        self._predict_step_outputs.append(results)

    def on_predict_epoch_end(self):
        numpy_results = self._evaluation_get_preds(self._predict_step_outputs)
        self._predict_step_outputs = []
        return numpy_results
