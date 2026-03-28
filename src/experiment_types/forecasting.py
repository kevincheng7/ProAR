from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

import src.utilities.metric as metric
from openfold.np import residue_constants
from openfold.np.protein import Protein, to_pdb
from openfold.utils.tensor_utils import dict_map
from src.diffusion.proar import ProAR
from src.experiment_types.forecasting_multi_horizon import AbstractMultiHorizonForecastingExperiment
from src.utilities.evaluation import evaluate_ensemble_prediction
from src.utilities.utils import tensor_to_ndarray, torch_to_numpy


class ProARForecasting(AbstractMultiHorizonForecastingExperiment):
    model: ProAR

    METRIC_SEP = "/forecast/"

    def __init__(self, save_dir: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.diffusion_config is not None, "diffusion config must be set. Use ``diffusion=<dyffusion>``!"
        assert self.diffusion_config.timesteps == self.horizon, "diffusion timesteps must be equal to horizon"
        if hasattr(self.model, "interpolator"):
            self.log_text.info(
                f"------------------- Setting num_predictions={getattr(self.hparams, 'num_predictions')}"
            )
            setattr(self.model.interpolator.hparams, "num_predictions", getattr(self.hparams, "num_predictions"))
        self.save_dir = save_dir

        # Exponential decay distribution for energy sampling
        prob_num = 500
        intensity = 2.0

        self.energy_grid = np.linspace(0, 1, prob_num)
        log_probs = intensity * self.energy_grid
        max_log_prob = np.max(log_probs)
        unnormalized_probs = np.exp(log_probs - max_log_prob)
        self.cdf = np.cumsum(unnormalized_probs)
        self.cdf /= self.cdf[-1]

    def use_stacked_ensemble_inputs(self, split) -> bool:
        return True  # always use stacked ensemble inputs

    def get_inputs_and_extra_kwargs_eval(
        self, batch: Dict, split: str, autoregressive_inputs: Optional[Dict] = None, ensemble: bool = True
    ) -> Tuple[Dict, Dict[str, Any]]:
        """
        Process validation/test/prediction batches with support for autoregression.

        Args:
            batch: Raw input data (shape depends on ensemble settings)
            autoregressive_inputs: Optional inputs for autoregressive steps

        Returns:
            Tuple of:
            - inputs: Model inputs (may be replaced by autoregressive_inputs)
            - extra_kwargs: Metadata and esm conditions
        """
        _ = batch.pop("time", None)
        metadata = batch.pop("meta_data", None)

        batch = self.get_ensemble_inputs(batch, split, add_noise=False)  # type: ignore

        inputs = {}
        extra_kwargs = {"metadata": metadata, "condition": {}}
        for key, value in batch.items():
            assert (
                value.shape[1] == self.window
            ), f"{split} data expected shape (b*num_predictions, {self.window}, *), got {value.shape}"

            past_steps = value[:, : self.window, ...]
            if self.window == 1:
                past_steps = past_steps.squeeze(1)

            if "esm" in key or "esm_pair" in key:
                extra_kwargs["condition"][key] = past_steps
            else:
                inputs[key] = past_steps

        if autoregressive_inputs is not None:
            inputs = autoregressive_inputs

        # Sample energy from exponential decay distribution
        rand_vals = np.random.rand(inputs["aatype"].shape[0])
        indices = np.searchsorted(self.cdf, rand_vals, side='right')
        energies = self.energy_grid[indices]
        extra_kwargs["condition"]["energy"] = torch.tensor(energies, device=self.device)

        return inputs, extra_kwargs

    # -----------------------------------
    # metrics with TorchMetrics
    # -----------------------------------
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        self.metric_name_to_class = {
            "ca_bond_dev": metric.CaBondDeviation,
            "ca_disconnect_rate": metric.CaDisconnectRate,
            "ca_steric_clash_rate": metric.CaStericClashRate,
            "num_ca_steric_clashes": metric.NumCaStericClashes,
        }
        metrics = {
            f"{split_name}/{self.horizon_name}_avg{self.METRIC_SEP}{key}": value().to(self.device)
            for key, value in self.metric_name_to_class.items()
        }
        for h in self.horizon_range:
            metrics.update(
                {
                    f"{split_name}/t{h}{self.METRIC_SEP}{key}": value().to(self.device)
                    for key, value in self.metric_name_to_class.items()
                }
            )
        return torch.nn.ModuleDict(metrics)

    # -----------------------------------
    #  evaluation with PyTorch Lightning
    # -----------------------------------
    def reshape_predictions(
        self,
        results: Dict[str, Dict[str, Tensor]],
        reshape_ensemble_dim: bool = True,
        add_ensemble_dim_in_inputs: bool = False,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Modified and simplified method which accept a nested dictionary as input.
        Reshape and unpack the predictions from the model. This modifies the input dictionary in-place.

        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
           reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
           add_ensemble_dim_in_inputs (bool, optional): Whether the ensemble dimension was added in the inputs.
        """
        ensemble_size = getattr(self.hparams, "num_predictions")
        for k, v in results.items():
            pred_keys = list(v.keys())
            preds_shape = v[pred_keys[0]].shape
            if reshape_ensemble_dim and preds_shape[0] > 1:
                if add_ensemble_dim_in_inputs or (ensemble_size > 1 and preds_shape[0] % ensemble_size == 0):
                    results[k] = self._reshape_ensemble_preds(results[k], "predict")
        return results

    def save_to_pdb(
        self,
        protein_coords: Tensor,
        features: Dict,
        t_step: Union[int, str],
        meta_data: List,
        suffix: str = "",
        save_dir: Optional[str] = None,
    ):
        """
        save a batch of structure to pdb file.
        Args:
            protein_coords (Tensor): (b, num_res, 37, 3).
            features (Dict): feature dict.
            t_step (int): the current step of interpolation.
            meta_data (list): used to infer the file name, of length (horizon + window)
                [{}, {}, {}, ... {}]
                with each {} being the metadata of that timestep, e.g.
                    {"system": ["1APV", "3DP4"], "time": [400, 500]} for batch size = 2
            suffix (str): suffix of the file name, could be "pred" or "tgt".
        """
        p_coords = tensor_to_ndarray(protein_coords)
        features = {k: tensor_to_ndarray(v) for k, v in features.items()}

        if save_dir is None:
            save_dir = getattr(self.hparams, "save_dir")
        assert save_dir is not None, "please specify save_dir during initialization"

        for b_idx, p_coord in enumerate(p_coords):
            system = meta_data[self.window - 1]["system"][b_idx]
            time = t_step

            p_file_name = f"{system}_{time}_{suffix}_protein.pdb"
            protein_save_dir = os.path.join(save_dir, f"{system}")
            os.makedirs(protein_save_dir, exist_ok=True)

            p_file_name = os.path.join(protein_save_dir, p_file_name)

            protein_obj = Protein(
                aatype=features["aatype"][b_idx],
                atom_positions=p_coord,
                atom_mask=features["atom37_atom_exists"][b_idx],
                residue_index=features["residue_index"][b_idx] + 1,
                b_factors=np.zeros_like(features["atom37_atom_exists"][b_idx]),
                chain_index=np.zeros_like(features["aatype"][b_idx]),
            )

            with open(p_file_name, "w") as fp:
                fp.write(to_pdb(protein_obj))

    def _eval_ensemble_predictions(self, outputs: List[Any], split: str):
        if not self.use_ensemble_predictions(split):
            return
        for output in outputs:
            all_preds_metrics = defaultdict(list)
            preds_key_metrics = defaultdict(list)
            ca = residue_constants.atom_order["CA"]

            for t_step in self.prediction_timesteps:
                t_preds = output[f"t{t_step}_preds"]
                t_targets = output[f"t{t_step}_targets"]

                p_preds = t_preds["all_atom_positions"][..., ca, :]  # (n, bs, num_res, 3)
                p_tgts = t_targets["all_atom_positions"][..., ca, :]  # (bs, num_res, 3)

                p_mask = np.expand_dims(t_targets["atom37_atom_exists"][..., ca], axis=-1)  # (n, bs, num_res)

                p_preds = np.where(p_mask, p_preds, 0)

                metrics = evaluate_ensemble_prediction(p_preds, p_tgts)

                for m, v in metrics.items():
                    preds_key_metrics[f"{split}/{self.ensemble_logging_infix(split)}t{t_step}/{m}"].append(v)
                    all_preds_metrics[f"{split}/{self.ensemble_logging_infix(split)}avg/{m}"].append(v)

        self.log_dict(
            {k: torch.tensor(np.mean(v), device=self.device) for k, v in preds_key_metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )

        avg_metrics = {k: torch.tensor(np.mean(v), device=self.device) for k, v in all_preds_metrics.items()}
        self.log_dict(avg_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def get_preds_at_t_for_batch(
        self,
        batch: Dict,
        horizon: int | float,
        split: str,
        autoregressive_inputs: Optional[Dict] = None,
        ensemble: bool = False,
        **kwargs,
    ) -> Dict:
        batch_shape = batch["aatype"].shape
        b, t = batch_shape[0:2]
        assert 0 < horizon <= self.true_horizon, f"horizon={horizon} must be in [1, {self.true_horizon}]"

        assert isinstance(
            self, ProARForecasting
        ), "get_preds_at_t_for_batch should only be called on a ProARForecasting instance"
        if horizon == self.prediction_timesteps[0]:
            if self.prediction_timesteps != self.horizon_range:
                setattr(self.model.hparams, "prediction_timesteps", [p_h for p_h in self.prediction_timesteps])
            inputs, extra_kwargs = self.get_inputs_and_extra_kwargs_eval(
                batch, split=split, autoregressive_inputs=autoregressive_inputs, ensemble=ensemble
            )
            if autoregressive_inputs is None:
                self._cached_start = dict_map(lambda x: x.detach().clone(), inputs, leaf_type=torch.Tensor)
            with torch.no_grad():
                self._current_preds = self.predict(inputs, x_start=self._cached_start, **extra_kwargs, **kwargs)

        preds_key = f"t{horizon}_preds"
        results = {k: self._current_preds.pop(k) for k in list(self._current_preds.keys()) if k == preds_key}
        if horizon == self.horizon_range[-1]:
            results = {**results, **self._current_preds}
            del self._current_preds
        return results

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: Optional[int] = None,
        return_outputs: bool | str = True,
        boundary_conditions: Optional[Callable] = None,
        t0: float = 0.0,
        dt: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Evaluation: use val data (sample), simulate the sampling process of the entire trajectory and evaluate it.
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results)
        Returns:
            results_dict: Dict[str, Tensor], where for each semantically different result, a separate prefix key is used
                Then, for each prefix key <p>, results_dict must contain <p>_preds and <p>_targets.
        """

        return_dict = dict()

        if return_outputs == "auto":
            return_outputs = True if split == "predict" else False
        log_dict = {"num_predictions": getattr(self.hparams, "num_predictions"), "noise_level": self.inputs_noise}

        save_to_pdb = split in ["test", "predict"]
        compute_metrics = split not in [
            "predict",
        ]
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None

        b = batch["aatype"].shape[0]
        meta_data = batch["meta_data"]

        if split == "val" and dataloader_idx in [0, None]:
            n_outer_loops = 1
        else:
            assert split in ["val", "test", "predict"]
            n_outer_loops = self.num_autoregressive_steps + 1

        if split_metrics is not None:
            avg_metric_keys = [
                f"{split}/{self.horizon_name}_avg{self.METRIC_SEP}{key}"
                for key in self.metric_name_to_class.keys()
            ]
            avg_metric_trackers = [split_metrics[key] for key in avg_metric_keys]
        else:
            avg_metric_keys = avg_metric_trackers = None

        autoregressive_inputs = None
        total_t = t0
        predicted_range_last = [0.0] + self.prediction_timesteps[:-1]
        ar_window_steps_t = self.horizon_range[-self.window :]
        _cumulative_network_s = 0.0
        _cumulative_relax_s = 0.0
        for ar_step in tqdm(
            range(n_outer_loops),
            desc="Autoregressive Step",
            position=0,
            leave=True,
            disable=not self.verbose or n_outer_loops <= 1,
        ):
            ar_window_steps = []
            for t_step_last, t_step in zip(predicted_range_last, self.prediction_timesteps):
                total_horizon = ar_step * self.true_horizon + t_step
                if total_horizon > self.prediction_horizon:
                    break

                pr_kwargs = {} if autoregressive_inputs is None else {"num_predictions": 1}
                if (int(ar_step) % 2) != 0:
                    pr_kwargs["use_x_start"] = True
                results = self.get_preds_at_t_for_batch(
                    batch, t_step, split, autoregressive_inputs, ensemble=True, **pr_kwargs
                )
                if t_step == self.prediction_timesteps[0]:
                    timing = self.model._last_timing
                    _cumulative_network_s += timing["network_s"]
                    _cumulative_relax_s += timing["relax_s"]
                total_t += dt * (t_step - t_step_last)
                total_horizon = ar_step * self.true_horizon + t_step

                preds = results.pop(f"t{t_step}_preds")
                if return_outputs in [True, "all"]:
                    return_dict[f"t{total_horizon}_preds"] = torch_to_numpy(preds)

                if return_outputs == "all":
                    return_dict.update(
                        {k.replace(f"t{t_step}", f"t{total_horizon}"): torch_to_numpy(v) for k, v in results.items()}
                    )

                if t_step in ar_window_steps_t:
                    if self.use_ensemble_predictions(split):
                        ar_window_steps += [
                            {
                                k: (
                                    v.reshape(v.shape[0] * v.shape[1])
                                    if v.dim() == 2
                                    else v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                                )
                                for k, v in preds.items()
                            }
                        ]
                    else:
                        ar_window_steps += [
                            preds,
                        ]

                if not compute_metrics or not float(total_horizon).is_integer():
                    continue

                pred_protein_coords = preds["all_atom_positions"]
                protein_mask = preds["all_atom_mask"]

                if self.use_ensemble_predictions(split):
                    pred_protein_coords = pred_protein_coords.mean(dim=0)

                if save_to_pdb and float(total_horizon).is_integer():
                    self.save_to_pdb(pred_protein_coords, preds, int(total_horizon), meta_data, suffix="pred")

                assert split_metrics is not None and avg_metric_trackers is not None
                metric_names = [
                    f"{split}/t{t_step}{self.METRIC_SEP}{key}" for key in self.metric_name_to_class.keys()
                ]
                for metric_name in metric_names:
                    metric = split_metrics[metric_name]
                    metric(
                        pred_protein_coords,
                        protein_mask,
                    )
                    log_dict[metric_name] = metric

                for tracker in avg_metric_trackers:
                    tracker(
                        pred_protein_coords,
                        protein_mask,
                    )

            if ar_step < n_outer_loops - 1:
                if self.window == 1:
                    autoregressive_inputs = ar_window_steps[0]
                else:
                    autoregressive_inputs = dict()
                    for k in ar_window_steps[0].keys():
                        autoregressive_inputs[k] = torch.stack(
                            [ar_window_step[k] for ar_window_step in ar_window_steps], dim=1
                        )
        if compute_metrics:
            assert avg_metric_trackers is not None and avg_metric_keys is not None
            log_kwargs = dict()
            log_kwargs["sync_dist"] = True
            for key, tracker in zip(avg_metric_keys, avg_metric_trackers):
                log_dict[key] = tracker
            self.log_dict(log_dict, on_step=False, on_epoch=True, **log_kwargs)

        return_dict["_timing"] = {
            "protein_name": meta_data[self.window - 1]["system"][0],
            "network_s": _cumulative_network_s,
            "relax_s": _cumulative_relax_s,
        }

        return return_dict

    def on_test_epoch_end(self, **kwargs) -> None:
        timing_rows = []
        for split, outputs in self._test_step_outputs.items():
            for out in outputs:
                if "_timing" in out:
                    t = out.pop("_timing")
                    timing_rows.append(t)

        super().on_test_epoch_end(**kwargs)

        if not timing_rows:
            return

        save_dir = getattr(self.hparams, "save_dir", None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "timing.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["protein_name", "network_s", "relax_s", "total_s"])
                writer.writeheader()
                for row in timing_rows:
                    row["total_s"] = row["network_s"] + row["relax_s"]
                    writer.writerow({k: f"{v:.3f}" if isinstance(v, float) else v for k, v in row.items()})

            total_network = sum(r["network_s"] for r in timing_rows)
            total_relax = sum(r["relax_s"] for r in timing_rows)
            total_all = sum(r["total_s"] for r in timing_rows)
            n = len(timing_rows)
            self.log_text.info(
                f"Timing summary ({n} proteins) → {csv_path}\n"
                f"  Network: {total_network:.1f}s total, {total_network / n:.2f}s avg\n"
                f"  Relax:   {total_relax:.1f}s total, {total_relax / n:.2f}s avg\n"
                f"  Total:   {total_all:.1f}s total, {total_all / n:.2f}s avg"
            )
