import inspect
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from torch import Tensor

import src.utilities.metric as metric
from openfold.np import residue_constants
from openfold.np.protein import Protein, to_pdb
from src.experiment_types._base_experiment import BaseExperiment
from src.models._base_model import BaseModel
from src.models.esmfold import ModifiedESMFoldInterpolation
from src.utilities.evaluation import evaluate_ensemble_prediction
from src.utilities.utils import tensor_to_ndarray


class InterpolationExperiment(BaseExperiment):
    r"""Interpolation experiment (inference-only)."""

    model: ModifiedESMFoldInterpolation

    def __init__(self, save_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"
        self.strict_loading = False

    def instantiate_model(self, *args, **kwargs) -> BaseModel:
        r"""Instantiate the model, e.g. by calling the constructor of the class :class:`BaseModel` or a subclass thereof."""

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
        self.log_text.info(f"Instantiated model: {model.__class__.__name__}")
        if self.is_diffusion_model:
            model = hydra.utils.instantiate(self.diffusion_config, model=model, _recursive_=False, **kwargs)
            self.log_text.info(
                f"Instantiated diffusion model: {model.__class__.__name__}, with"
                f" #diffusion steps={model.num_timesteps}"
            )
        return model

    @property
    def horizon_range(self) -> List[int]:
        return list(np.arange(1, self.horizon))  # type: ignore

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def METRIC_SEP(self) -> str:
        return "/ipol/"

    # --------------------------------- Metrics
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        self.metric_name_to_class = {
            "ca_bond_dev": metric.CaBondDeviation,
            "ca_disconnect_rate": metric.CaDisconnectRate,
            "ca_steric_clash_rate": metric.CaStericClashRate,
            "num_ca_steric_clashes": metric.NumCaStericClashes,
            "peptide_tm_score": metric.PeptideTmScore,
            "peptide_rmsd": metric.PeptideRMSD,
            "peptide_aligned_rmsd": metric.PeptideAlignedRMSD,
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

    @property
    def default_monitor_metric(self) -> str:
        return f"val/{self.horizon_name}_avg{self.METRIC_SEP}rmsd"

    def save_to_pdb(
        self,
        protein_coords: Tensor,
        features: Dict,
        t_step: int,
        meta_data: List,
        suffix: str = "",
    ):
        """
        Save a batch of structures to pdb file.

        Args:
            protein_coords (Tensor): (b, num_res, 37, 3).
            features (Dict): feature dict.
            t_step (int): the current step of interpolation.
            meta_data (list): of length (horizon + window),
                [{}, {}, {}, ... {}] with each {} being the metadata of that timestep.
            suffix (str): suffix of the file name, could be "pred" or "tgt".
        """
        p_coords = tensor_to_ndarray(protein_coords)
        features = {k: tensor_to_ndarray(v) for k, v in features.items()}

        save_dir = getattr(self.hparams, "save_dir")
        assert save_dir is not None, "please specify save_dir during initialization"

        for b_idx, p_coord in enumerate(p_coords):
            system = meta_data[0]["system"][b_idx]
            time = meta_data[0]["time"][b_idx]

            p_file_name = f"{system}_{time}_{t_step}_{suffix}_protein.pdb"
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

            for t_step in self.horizon_range:
                if not self.model.deterministic:
                    p_preds = output[f"t{t_step}_preds_final_atom_positions_uncrty"][..., ca, :]
                else:
                    p_preds = output[f"t{t_step}_preds_final_atom_positions"][..., ca, :]

                p_tgts = output[f"t{t_step}_targets_all_atom_positions"][..., ca, :]
                p_mask = np.expand_dims(
                    output[f"t{t_step}_preds_final_atom_mask"][..., ca], axis=-1
                )

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

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx=None,
        return_only_preds_and_targets: bool = False,
    ) -> Dict[str, Tensor]:
        """
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results).
        """
        log_dict = dict()
        compute_metrics = split == "val"
        save_to_pdb = split in ["test", "predict"]
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None
        b = batch["aatype"].shape[0]

        effective_batch_size = b * getattr(self.hparams, "num_predictions")

        return_dict = dict()
        if split_metrics is not None:
            avg_metric_keys = [
                f"{split}/{self.horizon_name}_avg{self.METRIC_SEP}{key}"
                for key in self.metric_name_to_class.keys()
            ]
            avg_metric_trackers = [split_metrics[key] for key in avg_metric_keys]
        else:
            avg_metric_keys = avg_metric_trackers = None
        inputs, meta_data = self.get_evaluation_inputs(batch, split=split)

        for t_step in self.horizon_range:
            targets = dict()
            for key, value in batch.items():
                targets[key] = value[torch.arange(b), 0, ...]
            aatype = targets["aatype"]
            protein_mask = targets["all_atom_mask"]

            time = torch.full((effective_batch_size,), t_step, device=self.device, dtype=torch.long)

            results = self.predict(inputs, time=time)
            if not self.model.deterministic:
                pred_protein_coords = results["preds_final_atom_positions_uncrty"]
            else:
                pred_protein_coords = results["preds_final_atom_positions"]

            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            return_dict = {**return_dict, **results}

            if self.use_ensemble_predictions(split):
                pred_protein_coords = pred_protein_coords.mean(dim=0)

            if save_to_pdb:
                self.save_to_pdb(pred_protein_coords, targets, t_step, meta_data, suffix="pred")

            if not compute_metrics:
                continue

        return return_dict

    def predict(
        self, inputs: Dict, num_predictions: Optional[int] = None, reshape_ensemble_dim: bool = True, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Main method for making predictions/doing inference.

        Args:
            inputs (dict): Input data dictionary.
            num_predictions (int, optional): Number of predictions to make. If None, use the default value.
            reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions in post-processed format.
        """
        base_num_predictions = getattr(self.hparams, "num_predictions")
        setattr(self.hparams, "num_predictions", num_predictions or base_num_predictions)
        if (
            hasattr(self.model, "sample_loop")
            and "num_predictions" in inspect.signature(getattr(self.model, "sample_loop")).parameters
        ):
            kwargs["num_predictions"] = getattr(self.hparams, "num_predictions")

        structure = self.model.predict_forward(inputs, **kwargs)
        post_process_keys = [
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
            "bb_update_weight",
            "frames_uncrty",
            "sidechain_frames_uncrty",
            "unnormalized_angles_uncrty",
            "angles_uncrty",
            "positions_uncrty",
            "bb_update_weight_uncrty",
        ]
        structure = {
            k: v.permute(*range(1, v.dim()), 0) if k in post_process_keys else v for k, v in structure.items()
        }

        results = {f"preds_{k}": v for k, v in structure.items()}

        setattr(self.hparams, "num_predictions", base_num_predictions)
        results = self.reshape_predictions(results, reshape_ensemble_dim)
        results = self.unpack_predictions(results)

        return results

    def get_evaluation_inputs(self, batch: Dict, split: str, **kwargs):
        """Get the network inputs from the dynamics data.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert split in [
            "val",
            "test",
            "predict",
        ], "split should be 'val', 'test' or 'predict', please assign the right split"
        meta_data = batch.pop("meta_data")
        _ = batch.pop("time", None)
        batch = self.get_ensemble_inputs(batch, split, add_noise=False)  # type: ignore
        inputs = dict()
        for key, value in batch.items():
            past_steps = value[:, : self.window, ...]
            if self.window == 1:
                past_steps = past_steps.squeeze(1)
            last_step = value[:, -1, ...]
            inputs[key] = [past_steps, last_step]
        return inputs, meta_data
