import os
import time as time_module
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from openfold.config import config
from openfold.data import data_transforms
from openfold.np import protein
from openfold.np.relax import relax
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import dict_map
from src.datamodules.datasets.mdtrajectory import process_protein_pdb
from src.datamodules.molecular_dynamics_simulation import length_batching, shape_schema
from src.diffusion.dyffusion import BaseDYffusion
from src.experiment_types.interpolation import InterpolationExperiment
from src.interface import get_checkpoint_from_path
from src.utilities import so3_utils
from src.utilities.hub import get_interpolator_checkpoint, get_interpolator_config
from src.utilities.utils import freeze_model, tensor_to_ndarray


TRANS_BASE_NOISE_SCALE = 10
ROTS_BASE_NOISE_SCALE = 1.5

COIL_TRANS_AMP_FACTOR = 1.5
COIL_ROTS_AMP_FACTOR = 1.2


class ProAR(BaseDYffusion):
    """ProAR diffusion model with a pretrained interpolator (inference-only)."""

    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        interpolator_local_checkpoint_path: Optional[str] = None,
        hydra_local_config_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["interpolator", "model"])
        ipol_ckpt = get_interpolator_checkpoint(interpolator_local_checkpoint_path)
        ipol_cfg = get_interpolator_config(hydra_local_config_path)
        self.interpolator: InterpolationExperiment = get_checkpoint_from_path(
            interpolator,
            ipol_ckpt,
            ipol_cfg,
        )
        freeze_model(self.interpolator)

        self.interpolator_window = self.interpolator.window
        self.interpolator_horizon = self.interpolator.true_horizon
        last_d_to_i_tstep = self.diffusion_step_to_interpolation_step(self.num_timesteps - 1)
        if self.interpolator_horizon != last_d_to_i_tstep + 1:
            raise ValueError(
                f"interpolator horizon {self.interpolator_horizon} must be equal to the "
                f"last interpolation step+1=i_N=i_{self.num_timesteps - 1}={last_d_to_i_tstep + 1}"
            )

        sigma_grid = torch.linspace(0.1, 1.8, 1000)
        self.igso3 = so3_utils.SampleIGSO3(
            1000, sigma_grid, cache_dir=os.path.expanduser("~/.cache/proar/igso3")
        )

        self._relax_config = config.relax
        self._amber_relaxer = None
        self._last_timing = {"network_s": 0.0, "relax_s": 0.0}

    @property
    def amber_relaxer(self):
        if self._amber_relaxer is None:
            self._amber_relaxer = relax.AmberRelaxation(
                use_gpu=(str(self.device) != "cpu"),
                **self._relax_config,
            )
        return self._amber_relaxer

    def corrupt_rigid(self, x_t: Dict, t: Tensor):
        """Add noise to rigid body transformations for the sampling process."""
        rigid_t = Rigid.from_tensor_4x4(x_t["backbone_rigid_tensor"])
        trans_t = rigid_t.get_trans()
        rots_t = rigid_t.get_rots().get_rot_mats()
        mask = x_t["backbone_rigid_mask"]
        dssp = x_t["dssp"]

        is_coil = dssp == 3
        coil_mask = is_coil.unsqueeze(-1)
        coil_mask_rot = coil_mask.unsqueeze(-1)

        t_flow = 0.1 + (t / (self.num_timesteps - 1)) * (0.2 - 0.1)
        downscale_fac = 1.0 - t_flow

        b, n = trans_t.shape[:2]
        noise_trans = torch.randn(b, n, 3, device=self.device, dtype=trans_t.dtype)

        amplified_noise_trans = noise_trans * COIL_TRANS_AMP_FACTOR
        noise_trans = torch.where(coil_mask, amplified_noise_trans, noise_trans)

        corrupted_trans = TRANS_BASE_NOISE_SCALE * downscale_fac[..., None, None] * noise_trans + trans_t
        corrupted_trans = corrupted_trans - torch.mean(corrupted_trans, dim=-2, keepdim=True)
        corrupted_trans = corrupted_trans * mask[..., None]

        base_noise_rots = (
            torch.stack(
                [
                    self.igso3.sample(
                        torch.tensor(
                            [ROTS_BASE_NOISE_SCALE * downscale_fac[i]], device=self.device, dtype=rots_t.dtype
                        ),
                        n,
                    )
                    for i in range(b)
                ]
            )
            .to(self.device, dtype=rots_t.dtype)
            .reshape(b, n, 3, 3)
        )

        amp_noise_rots = (
            torch.stack(
                [
                    self.igso3.sample(
                        torch.tensor(
                            [ROTS_BASE_NOISE_SCALE * COIL_ROTS_AMP_FACTOR * downscale_fac[i]],
                            device=self.device,
                            dtype=rots_t.dtype,
                        ),
                        n,
                    )
                    for i in range(b)
                ]
            )
            .to(self.device, dtype=rots_t.dtype)
            .reshape(b, n, 3, 3)
        )
        noise_rots = torch.where(coil_mask_rot, amp_noise_rots, base_noise_rots)

        corrupted_rots = torch.einsum("...ij,...jk->...ik", rots_t, noise_rots)
        identity = torch.eye(3, device=rots_t.device)
        corrupted_rots = corrupted_rots * mask[..., None, None] + identity[None, None] * (1 - mask[..., None, None])

        x_t["backbone_rigid_tensor"] = Rigid(
            rots=Rotation(rot_mats=corrupted_rots), trans=corrupted_trans
        ).to_tensor_4x4()

        return x_t, t_flow

    def _interpolate(self, initial_condition: Dict, x_last: Dict, t: Tensor, static_condition: Dict, **kwargs) -> Dict:
        assert (0 < t).all() and (
            t < self.interpolator_horizon
        ).all(), f"interpolate time must be in (0, {self.interpolator_horizon}), got {t}"

        initial_condition = {**initial_condition, **static_condition}
        x_last = {**x_last, **static_condition}
        interpolator_inputs = dict()
        for key in initial_condition.keys():
            assert key in x_last.keys(), "initial_condition and x_last does not match."
            interpolator_inputs[key] = [initial_condition[key], x_last[key]]

        kwargs["reshape_ensemble_dim"] = False
        interpolator_outputs = self.interpolator.predict(
            interpolator_inputs, condition=static_condition, time=t, **kwargs
        )
        return self.post_process_output(
            interpolator_outputs, is_interpolator=True, use_deterministic=kwargs.get("use_deterministic", True)
        )

    def select(self, input_dict: Dict[str, Tensor], mask) -> Dict:
        """Filter values in input_dict using a given mask."""
        ret = dict()
        for key, value in input_dict.items():
            assert torch.is_tensor(value), "only support tensor selection."
            ret[key] = value[mask]
        return ret

    def post_process_output(self, structure: Dict, is_interpolator=False, use_deterministic=True) -> Dict:
        """Post-process the output dict so it can be fed into a subsequent network."""
        prefix = "preds_" if is_interpolator else ""
        suffix = "_uncrty" if is_interpolator and not self.interpolator.model.deterministic else ""
        ret = {
            "backbone_rigid_tensor": Rigid.from_tensor_7(
                structure[f"{prefix}final_affine_tensor{suffix}"]
            ).to_tensor_4x4(),
            "all_atom_positions": structure[f"{prefix}final_atom_positions{suffix}"],
        }
        return ret

    def get_condition(
        self,
        condition,
        x_last: Optional[Dict],
        prediction_type: str,
        static_condition: Optional[Dict] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> Tuple[Dict, Dict]:
        """Simply returns the two conditions."""
        assert static_condition is not None, "please specify static_condition."
        return condition, static_condition

    def adapted_cold_sampling_iter(self, x_s, x_interpolated_s, x_interpolated_s_next) -> Dict:
        x_s["all_atom_positions"] = (
            x_s["all_atom_positions"]
            - x_interpolated_s["all_atom_positions"]
            + x_interpolated_s_next["all_atom_positions"]
        )

        def transform_fns():
            transforms = [
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
            return transforms

        @data_transforms.curry1
        def compose(x, fs):
            for f in fs:
                x = f(x)
            return x

        transform = transform_fns()
        x_s = compose(transform)(x_s)
        return x_s

    def relax(self, x: Dict):
        """Relax protein structures using AMBER relaxer."""
        batch_size = x["aatype"].shape[0]
        x_np = dict_map(lambda x: tensor_to_ndarray(x), x, leaf_type=torch.Tensor)

        x_relaxed = []
        for b in range(batch_size):
            protein_obj_unrelaxed = protein.Protein(
                aatype=x_np["aatype"][b],
                atom_positions=x_np["all_atom_positions"][b],
                atom_mask=x_np["atom37_atom_exists"][b],
                residue_index=x_np["residue_index"][b] + 1,
                b_factors=np.zeros_like(x_np["atom37_atom_exists"][b]),
                chain_index=np.zeros_like(x_np["aatype"][b]),
            )

            struct_str, debug_data, violations = self.amber_relaxer.process(prot=protein_obj_unrelaxed)
            x_relaxed.append(process_protein_pdb(pdb_str=struct_str))

        x_relaxed = length_batching(x_relaxed, shape_schema, use_length_batching=False)
        x_relaxed = dict_map(lambda x: x.to(self.device), x_relaxed, leaf_type=torch.Tensor)

        x.update(x_relaxed)

        return x

    def sample_loop(
        self,
        initial_condition: Dict,
        x_start: Dict,
        static_condition: Optional[Dict] = None,
        log_every_t: Optional[Union[str, int]] = None,
        num_predictions: Optional[int] = None,
        use_x_start: bool = False,
    ):
        """
        Args:
            initial_condition: the first frame of the current AR step,
                values are of shape (b * num_predictions, n, *).
            x_start: the first frame of the trajectory,
                values are of shape (b * num_predictions, n, *).
            static_condition: the static condition data with key "esm",
                values are of shape (b * num_predictions, n, *)
        """
        assert static_condition is not None, "please specify static_condition."
        cond_type = getattr(self.hparams, "forward_conditioning")
        assert (
            "data+noise" not in cond_type and "none" not in cond_type
        ), "currently ProAR only supports 'data' condition."

        batch_size = initial_condition["aatype"].size(dim=0)
        log_every_t = log_every_t or getattr(self.hparams, "log_every_t")
        log_every_t = log_every_t if log_every_t != "auto" else 1

        sc_kw = dict(static_condition=static_condition)
        x_s = dict_map(lambda x: x.detach().clone(), initial_condition, leaf_type=torch.Tensor)
        intermediates, x0_hat, dynamics_pred_step = dict(), None, 0
        last_i_n_plus_one = self.sampling_schedule[-1] + 1
        s_and_snext = zip(
            self.sampling_schedule,
            self.sampling_schedule[1:] + [last_i_n_plus_one],
        )

        _relax_time = 0.0
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        _t_start = time_module.perf_counter()

        for s, s_next in tqdm(
            s_and_snext, desc="Sampling time step", total=len(self.sampling_schedule), leave=False
        ):
            is_last_step = s == self.num_timesteps - 1

            step_s = torch.full((batch_size,), s, dtype=torch.float32, device=self.device)
            x0_hat = dict_map(lambda x: x.detach().clone(), initial_condition, leaf_type=torch.Tensor)

            if use_x_start or (int(s) % 2) != 0:
                x_corrupted = dict_map(lambda x: x.detach().clone(), x_start, leaf_type=torch.Tensor)
                x_corrupted, step_s_flow = self.corrupt_rigid(x_corrupted, step_s)
                structure = self.predict_x_last(
                    condition=x_s, x_t=x_corrupted, t=step_s_flow, is_sampling=True, **sc_kw
                )
            else:
                x_corrupted = dict_map(lambda x: x.detach().clone(), x_s, leaf_type=torch.Tensor)
                x_corrupted, step_s_flow = self.corrupt_rigid(x_corrupted, step_s)
                structure = self.predict_x_last(
                    condition=initial_condition, x_t=x_corrupted, t=step_s_flow, is_sampling=True, **sc_kw
                )

            x0_hat.update(self.post_process_output(structure))

            time_i_n = self.diffusion_step_to_interpolation_step(s_next) if not is_last_step else np.inf
            is_dynamics_pred = float(time_i_n).is_integer() or is_last_step
            q_sample_kwargs = dict(
                x0=x0_hat,
                x_end=initial_condition,
                is_artificial_step=not is_dynamics_pred,
                reshape_ensemble_dim=not is_last_step,
                num_predictions=1 if is_last_step else num_predictions,
            )
            if s_next <= self.num_timesteps - 1:
                step_s_next = torch.full((batch_size,), s_next, dtype=torch.float32, device=self.device)
                x_interpolated_s_next = dict_map(
                    lambda x: x.detach().clone(), initial_condition, leaf_type=torch.Tensor
                )
                x_interpolated = self.q_sample(**q_sample_kwargs, t=step_s_next, **sc_kw)
                x_interpolated_s_next.update(x_interpolated)
            else:
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                _t_relax_start = time_module.perf_counter()
                x0_hat = self.relax(x0_hat)
                _relax_time += time_module.perf_counter() - _t_relax_start
                x_interpolated_s_next = x0_hat

            if getattr(self.hparams, "sampling_type") in ["cold"]:
                if s > 0:
                    x_interpolated_s = dict_map(
                        lambda x: x.detach().clone(), initial_condition, leaf_type=torch.Tensor
                    )
                    x_interpolated = self.q_sample(**q_sample_kwargs, t=step_s, **sc_kw)
                    x_interpolated_s.update(x_interpolated)
                else:
                    x_interpolated_s = x_s
                x_s = self.adapted_cold_sampling_iter(x_s, x_interpolated_s, x_interpolated_s_next)
            elif getattr(self.hparams, "sampling_type") == "naive":
                x_s = x_interpolated_s_next
            else:
                raise ValueError(f"unknown sampling type {getattr(self.hparams, 'sampling_type')}")

            dynamics_pred_step = int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
            if is_dynamics_pred:
                intermediates[f"t{dynamics_pred_step}_preds"] = x_s
                if log_every_t is not None:
                    intermediates[f"t{dynamics_pred_step}_preds2"] = x_interpolated_s_next

            s1, s2 = s, s
            if log_every_t is not None:
                intermediates[f"intermediate_{s1}_x0hat"] = x0_hat
                intermediates[f"xipol_{s2}_dmodel"] = x_interpolated_s_next
                if getattr(self.hparams, "sampling_type") == "cold":
                    intermediates[f"xipol_{s1}_dmodel2"] = x_interpolated_s

        if getattr(self.hparams, "refine_intermediate_predictions"):
            q_sample_kwargs["x0"] = x0_hat
            q_sample_kwargs["is_artificial_step"] = False
            dynamical_steps = getattr(self.hparams, "prediction_timesteps") or list(self.dynamical_steps.values())
            dynamical_steps = [i for i in dynamical_steps if i < self.num_timesteps]
            sc_kw["use_deterministic"] = False
            for i_n in dynamical_steps:
                i_n_time_tensor = torch.full((batch_size,), float(i_n), dtype=torch.float32, device=self.device)
                i_n_for_str = int(i_n) if float(i_n).is_integer() else i_n
                assert (
                    not float(i_n).is_integer() or f"t{i_n_for_str}_preds" in intermediates
                ), f"t{i_n_for_str}_preds not in intermediates"
                intermediates[f"t{i_n_for_str}_preds"] = dict_map(
                    lambda x: x.detach().clone(), initial_condition, leaf_type=torch.Tensor
                )
                x_interpolated = self.q_sample(
                    **q_sample_kwargs, t=None, interpolation_time=i_n_time_tensor, **sc_kw
                )
                intermediates[f"t{i_n_for_str}_preds"].update(x_interpolated)

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        _t_total = time_module.perf_counter() - _t_start
        self._last_timing = {
            "network_s": _t_total - _relax_time,
            "relax_s": _relax_time,
        }

        if last_i_n_plus_one < self.num_timesteps:
            return x_s, intermediates, x_interpolated_s_next
        return x0_hat, intermediates, x_s
