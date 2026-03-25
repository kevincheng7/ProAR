import logging
from typing import Dict, Optional

import ml_collections
import torch
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf
from torch import nn

from openfold.config import config
from openfold.np import residue_constants
from openfold.utils.loss import (
    backbone_loss,
    compute_renamed_ground_truth,
    distogram_loss,
    find_structural_violations,
    lddt,
    sidechain_loss,
    softmax_cross_entropy,
    supervised_chi_loss,
    violation_loss,
)
from openfold.utils.rigid_utils import Rigid, Rotation
from src.utilities.utils import batch_align_structures


def matrix_to_axis_angle(matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.
    Ref: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L507

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    # if not fast:
    #     return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = 0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (matrix[near_pi][..., 0, :] + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device))
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles


def merge_config_dict(base: ConfigDict, *others) -> ConfigDict:
    """
    Recursively merges multiple dict or ConfigDict instances into the base ConfigDict.
    If keys overlap, values from later dictionaries overwrite earlier ones.

    Args:
        base (ConfigDict): The base ConfigDict to merge into.
        *others: Additional dictionaries or ConfigDict instances to merge.

    Returns:
        ConfigDict: A new merged ConfigDict.
    """

    def recursive_merge(dest: ConfigDict, src: dict):
        for key, value in src.items():
            if isinstance(value, dict):
                if key not in dest:
                    dest[key] = ConfigDict()
                elif not isinstance(dest[key], (dict, ConfigDict)):
                    raise TypeError(f"Cannot merge non-dict value into dict at key '{key}'")
                recursive_merge(dest[key], value)  # type: ignore
            else:
                dest[key] = value

    merged = ConfigDict(base.to_dict())  # Create a deep copy to avoid modifying original
    for other in others:
        if not isinstance(other, (dict, ConfigDict)):
            raise TypeError(f"Expected dict or ConfigDict, but got {type(other)}")
        recursive_merge(merged, other.to_dict() if isinstance(other, ConfigDict) else other)

    return merged


def protein_aa_mse_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    pred_pos = out["positions"][-1]  # (bs, num_res, 14, 3)
    gt_pos = batch["atom14_gt_positions"]
    mask = batch["atom14_gt_exists"]  # (bs, num_res, 14)
    b = mask.shape[0]

    pred_pos, gt_pos, _ = batch_align_structures(
        pred_pos.reshape(b, -1, 3), gt_pos.reshape(b, -1, 3), mask.reshape(b, -1)
    )
    pred_pos = pred_pos.reshape(b, -1, 14, 3)
    gt_pos = gt_pos.reshape(b, -1, 14, 3)

    protein_diff = (pred_pos - gt_pos) ** 2  # (batch_size, num_residue, 14, 3)
    mask_expanded = mask.unsqueeze(-1)  # (batch_size, num_residue, 14, 1)
    protein_diff = protein_diff * mask_expanded

    protein_mask_sum = mask_expanded.sum()
    protein_mse_loss = protein_diff.sum() / protein_mask_sum.clamp(min=1e-5)

    return protein_mse_loss


def mle_loss_chroma(
    L: torch.Tensor,  # (bs, n_res, n_res)
    pred_aa_pos: torch.Tensor,  # (bs, n_res, 37, 3)
    batch: Dict[str, torch.Tensor],
):
    ca = residue_constants.atom_order["CA"]

    # Ensure NLL gradients don't train the mean prediction module
    pred_aa_pos_detached = pred_aa_pos.detach()

    gt_ca_pos = batch["all_atom_positions"][..., ca, :]  # (bs, n_res, 3)
    pred_ca_pos_detached = pred_aa_pos_detached[..., ca, :]  # (bs, n_res, 3)
    bs, n = pred_ca_pos_detached.shape[:2]

    negs = []
    for b in range(bs):
        # (n_res,) => (n_res, 1) => (n_res, 3) => (3, num_res)
        mask_p = batch["all_atom_mask"][b, :, ca].unsqueeze(-1).expand(-1, 3).permute([1, 0])  # (3, n_res)
        len_p = mask_p[0].sum().to(torch.long)

        pred_ca = pred_ca_pos_detached[b].reshape(-1, 3)  # (n_res, 3)
        pred_ca = pred_ca.permute([1, 0])  # (3, n_res)
        gt_ca = gt_ca_pos[b].reshape(-1, 3)  # (n_res, 3)
        gt_ca = gt_ca.permute([1, 0])  # (3, n_res)
        # (n_res, n_res) => (n_res, n_res, 1) => (n_res, n_res, 3) => (3, n_res, n_res)
        scale_tril = L[b].unsqueeze(-1).expand(-1, -1, 3).permute([2, 0, 1])

        try:
            distribution_p = torch.distributions.MultivariateNormal(
                pred_ca[..., :len_p], scale_tril=scale_tril[..., :len_p, :len_p].float()
            )
            neg_log_likelihood_p = -distribution_p.log_prob(gt_ca[..., :len_p])
        except torch._C._LinAlgError:  # type: ignore
            neg_log_likelihood_p = torch.tensor(0.0, device=pred_ca.device)
            logging.warning("Cholesky factorization failed.")

        neg_log_likelihood = torch.mean(neg_log_likelihood_p)
        negs.append(neg_log_likelihood)

    return torch.stack(negs).mean()


def variance_penalty_loss(
    L: torch.Tensor,  # (bs, n_res, n_res)
    ca_variance: torch.Tensor,  # (bs, n_res, 3)
):
    bs, N, _ = L.shape

    tril_mask = torch.tril(torch.ones((N, N), device=L.device))  # (N, N)
    tril_mask = tril_mask.unsqueeze(0).expand(bs, -1, -1)  # (bs, N, N)

    var = (L**2 * tril_mask).sum(dim=2)  # (bs, N)

    var_xyz = var.unsqueeze(-1).expand(-1, -1, 3)  # (bs, N, 3)

    valid_mask = ca_variance > 0  # (bs, N, 3)

    delta = torch.relu(var_xyz - ca_variance)
    penalty = (delta**2) * valid_mask

    return penalty.sum()


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps)

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (eps + torch.sum(all_atom_mask, dim=-1))

    # loss = loss * (
    #     (resolution >= min_resolution) & (resolution <= max_resolution)
    # )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def smooth_lddt_loss(
    pred_aa_pos: torch.Tensor,  # L, b, n, 14, 3
    gt_aa_pos: torch.Tensor,  # b, n, 37, 3
    is_dna: Optional[torch.Tensor] = None,  # b, n
    is_rna: Optional[torch.Tensor] = None,  # b, n
    aa_mask: Optional[torch.Tensor] = None,  # b, n, 37
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    lddt_thresholds: torch.Tensor = torch.tensor([0.5, 1.0, 2.0, 4.0]),
) -> torch.Tensor:
    """
    Compute Smooth LDDT Loss.
    ref: https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/alphafold3.py#L3140

    Args:
        pred_aa_pos: Predicted atomic coordinates (L, b, n, 14, 3)
        gt_aa_pos: Ground truth atomic coordinates (b, n, 37, 3)
        is_dna: Boolean tensor indicating DNA atoms (b, n)
        is_rna: Boolean tensor indicating RNA atoms (b, n)
        aa_mask: Mask tensor for valid coordinates (b, n, 37)
        nucleic_acid_cutoff: Distance cutoff for nucleic acids
        other_cutoff: Distance cutoff for other atoms
        lddt_thresholds: LDDT thresholds (4,)

    Returns:
        Smooth LDDT loss (scalar tensor)
    """
    lddt_thresholds = lddt_thresholds.to(pred_aa_pos.device)
    b, n = gt_aa_pos.shape[:2]

    ca = residue_constants.atom_order["CA"]
    pred_ca_coords = pred_aa_pos[..., ca, :]  # (L, b, n, 3)
    gt_ca_coords = gt_aa_pos[..., ca, :]  # (b, n, 3)

    # Compute pairwise distances
    true_dists = torch.cdist(gt_ca_coords, gt_ca_coords)  # (b, n, n)
    pred_dists = torch.cdist(pred_ca_coords, pred_ca_coords)  # (L, b, n, n)

    # Compute distance differences
    dist_diff = torch.abs(true_dists.unsqueeze(0) - pred_dists)  # (L, b, n, n)

    # Compute epsilon values
    eps = lddt_thresholds.view(1, 1, 1, 1, -1) - dist_diff.unsqueeze(-1)  # (L, b, n, n, 4)
    eps = eps.sigmoid().mean(dim=-1)  # (L, b, n, n)

    # Restrict to bespoke inclusion radius
    if is_dna is not None and is_rna is not None:
        is_nucleotide = is_dna | is_rna  # (b, n)
    else:
        is_nucleotide = torch.zeros(b, n, device=pred_aa_pos.device, dtype=torch.bool)

    is_nucleotide_pair = is_nucleotide.unsqueeze(1) & is_nucleotide.unsqueeze(2)  # (b, n, n)
    inclusion_radius = torch.where(
        is_nucleotide_pair, true_dists < nucleic_acid_cutoff, true_dists < other_cutoff
    )  # (b, n, n)

    # Compute mask (avoid self-comparison)
    mask = inclusion_radius & ~torch.eye(gt_ca_coords.shape[1], dtype=torch.bool, device=gt_ca_coords.device)
    mask = mask.unsqueeze(0)  # Expand for L layers (1, b, n, n)

    if aa_mask is not None:
        ca_mask = aa_mask[..., ca].to(torch.bool)  # (b, n)
        coords_mask_pair = ca_mask.unsqueeze(1) & ca_mask.unsqueeze(2)
        mask = mask & coords_mask_pair.unsqueeze(0)

    # Compute masked average
    lddt = (eps * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1e-6)  # (L, b)

    return 1.0 - lddt.mean()


def pairwise_dist_loss(
    pred_aa_pos: torch.Tensor,  # L, b, n, 14, 3
    gt_aa_pos: torch.Tensor,  # b, n, 37, 3
    aa_mask: torch.Tensor,  # b, n, 37
) -> torch.Tensor:
    b, n = gt_aa_pos.shape[:2]

    ca = residue_constants.atom_order["CA"]
    pred_ca_coords = pred_aa_pos[..., ca, :]  # (L, b, n, 3)
    gt_ca_coords = gt_aa_pos[..., ca, :]  # (b, n, 3)

    # Compute pairwise distances
    true_dists = torch.cdist(gt_ca_coords, gt_ca_coords)  # (b, n, n)
    pred_dists = torch.cdist(pred_ca_coords, pred_ca_coords)  # (L, b, n, n)

    # Compute distance differences
    dist_diff = torch.abs(true_dists.unsqueeze(0) - pred_dists)  # (L, b, n, n)

    ca_mask = aa_mask[..., ca].to(torch.bool)  # (b, n)
    coords_mask_pair = ca_mask.unsqueeze(1) & ca_mask.unsqueeze(2)

    loss = (dist_diff * coords_mask_pair).sum(dim=(-1, -2)) / (coords_mask_pair.sum(dim=(-1, -2)) + 1e-6)

    return loss.mean()


def neighbor_ca_dist_loss(
    pred_aa_pos: torch.Tensor,  # L, b, n, 14, 3
    gt_aa_pos: torch.Tensor,  # b, n, 37, 3
    aa_mask: torch.Tensor,  # b, n, 37
    std: Optional[torch.Tensor] = None,  # b, n-1 (additional std value for each pair of CA)
) -> torch.Tensor:
    b, n = gt_aa_pos.shape[:2]

    ca = residue_constants.atom_order["CA"]
    pred_ca_coords = pred_aa_pos[..., ca, :]  # (L, b, n, 3)
    gt_ca_coords = gt_aa_pos[..., ca, :]  # (b, n, 3)
    ca_mask = aa_mask[..., ca].to(torch.bool)  # (b, n)

    pred_diff = torch.diff(pred_ca_coords, dim=-2)  # (L, b, n-1, 3)
    gt_diff = torch.diff(gt_ca_coords, dim=-2)  # (b, n-1, 3)

    pred_dist = torch.norm(pred_diff, dim=-1)  # (L, b, n-1)
    gt_dist = torch.norm(gt_diff, dim=-1)  # (b, n-1)

    dist_diff = torch.abs(gt_dist.unsqueeze(0) - pred_dist)  # (L, b, n-1)
    valid_mask = ca_mask[:, :-1] & ca_mask[:, 1:]  # (b, n-1)
    valid_mask = valid_mask.unsqueeze(0).float()  # (1, b, n-1)

    if std is not None:
        dist_diff = dist_diff * torch.clamp((1 / std + 1e-6), max=100.0).unsqueeze(0)  # (L, b, n-1)

    loss = (dist_diff * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-6)

    return loss.mean()


def rotation_mse_loss(
    out,
    batch,
):
    # rigids with 7 final colums, four for the quaternion followed by three for the translation.
    frames = out["frames"]  # L, b, n, 7
    gt_frames = batch["backbone_rigid_tensor"]  # b, n, 4, 4
    mask = batch["backbone_rigid_mask"]  # b, n

    rots = Rotation(quats=frames[..., :4]).get_rot_mats()
    gt_rots = Rigid.from_tensor_4x4(gt_frames).get_rots().get_rot_mats()

    rot_vecs = matrix_to_axis_angle(rots)
    gt_rot_vecs = matrix_to_axis_angle(gt_rots)

    mask_expanded = mask.unsqueeze(-1)  # (b, n, 1)
    vec_diff = (rot_vecs - gt_rot_vecs) ** 2  # (L, b, n, 3)
    vec_diff_masked = vec_diff * mask_expanded  # (L, B, n, 3)

    mask_sum = mask_expanded.sum() * vec_diff_masked.shape[0]
    vec_mse_loss = vec_diff_masked.sum() / mask_sum.clamp(min=1e-5)

    return vec_mse_loss


def rotation_mse_loss_uncrty(
    out,
    batch,
):
    # rigids with 7 final colums, four for the quaternion followed by three for the translation.
    frames = out["frames_uncrty"]  # L, b, n, 7
    gt_frames = batch["backbone_rigid_tensor"]  # b, n, 4, 4
    mask = batch["backbone_rigid_mask"]  # b, n

    rots = Rotation(quats=frames[..., :4]).get_rot_mats()
    gt_rots = Rigid.from_tensor_4x4(gt_frames).get_rots().get_rot_mats()

    rot_vecs = matrix_to_axis_angle(rots)
    gt_rot_vecs = matrix_to_axis_angle(gt_rots)

    mask_expanded = mask.unsqueeze(-1)
    vec_diff = (rot_vecs - gt_rot_vecs) ** 2  # (L, b, n, 3)
    vec_diff_masked = vec_diff * mask_expanded  # (L, b, n, 3)

    mask_sum = mask_expanded.sum() * vec_diff_masked.shape[0]
    vec_mse_loss = vec_diff_masked.sum() / mask_sum.clamp(min=1e-5)

    return vec_mse_loss


def bb_update_loss(
    linear_weight,
    clamp_value: float = 5.0,
):
    return -torch.clamp(torch.sum(linear_weight**2), min=0, max=clamp_value)


def fape_loss(
    out: Dict,
    batch: Dict,
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    bb_loss = backbone_loss(
        traj=out["frames"],
        residue_variance=torch.mean(batch["ca_variance"], dim=-1),
        **{**batch, **getattr(config, "backbone")},
    )

    sc_loss = sidechain_loss(
        out["sidechain_frames"],
        out["positions"],
        residue_variance=torch.mean(batch["ca_variance"], dim=-1),
        **{**batch, **getattr(config, "sidechain")},
    )

    loss = getattr(config, "backbone.weight") * bb_loss + getattr(config, "sidechain.weight") * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def fape_loss_uncrty(
    out: Dict,
    batch: Dict,
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    bb_loss = backbone_loss(
        traj=out["frames_uncrty"],
        **{**batch, **getattr(config, "backbone")},
    )

    sc_loss = sidechain_loss(
        out["sidechain_frames_uncrty"],
        out["positions_uncrty"],
        **{**batch, **getattr(config, "sidechain")},
    )

    loss = getattr(config, "backbone.weight") * bb_loss + getattr(config, "sidechain.weight") * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, custom_config: DictConfig):
        super().__init__()
        assert (
            "deterministic" in custom_config
        ), f"Please specify `deterministic` in custom_config: {custom_config} for loss computation."
        self.use_uncertainty_loss = not custom_config.pop("deterministic")

        self.config = config.loss

        # uncertainty related loss functions
        if self.use_uncertainty_loss:
            self.config.violation_uncrty = self.config.violation  # type: ignore
            self.config.fape_uncrty = self.config.fape  # type: ignore
            self.config.supervised_chi_uncrty = self.config.supervised_chi  # type: ignore

        self.config = merge_config_dict(self.config, OmegaConf.to_container(custom_config))  # type: ignore

    def forward(self, out, batch, _return_breakdown=True):
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["positions"][-1],
                **getattr(self.config, "violation"),
            )
        if "violation_uncrty" not in out.keys() and self.use_uncertainty_loss:
            out["violation_uncrty"] = find_structural_violations(
                batch,
                out["positions_uncrty"][-1],
                **getattr(self.config, "violation"),
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **getattr(self.config, "distogram")},
            ),
            "smooth_lddt": lambda: smooth_lddt_loss(
                out["positions"],
                batch["all_atom_positions"],
                aa_mask=batch["all_atom_mask"],
            ),
            "neighbor_ca_dist": lambda: neighbor_ca_dist_loss(
                out["positions"], batch["all_atom_positions"], batch["all_atom_mask"], std=batch.get("neighbor_ca_std")
            ),
            "pairwise_dist": lambda: pairwise_dist_loss(
                out["positions"],
                batch["all_atom_positions"],
                batch["all_atom_mask"],
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                getattr(self.config, "fape"),
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **getattr(self.config, "plddt_loss")},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["angles"],
                out["unnormalized_angles"],
                **{**batch, **getattr(self.config, "supervised_chi")},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **batch,
            ),
            "protein_mse": lambda: protein_aa_mse_loss(
                out,
                batch,
            ),
            "rotation_mse": lambda: rotation_mse_loss(
                out,
                batch,
            ),
        }
        if "bb_update_weight" in out.keys():
            loss_fns["bb_update"] = lambda: bb_update_loss(out["bb_update_weight"])
        # ------------------ uncertainty related loss functions
        if self.use_uncertainty_loss:
            loss_fns.update(
                {
                    "mle": lambda: mle_loss_chroma(
                        out["scale_tril_p"],
                        out["final_atom_positions"],
                        batch,
                    ),
                    "variance_penalty": lambda: variance_penalty_loss(
                        out["scale_tril_p"],
                        batch["ca_variance"],
                    ),
                    "smooth_lddt_uncrty": lambda: smooth_lddt_loss(
                        out["positions_uncrty"],
                        batch["all_atom_positions"],
                        aa_mask=batch["all_atom_mask"],
                    ),
                    "pairwise_dist_uncrty": lambda: pairwise_dist_loss(
                        out["positions_uncrty"],
                        batch["all_atom_positions"],
                        batch["all_atom_mask"],
                    ),
                    "fape_uncrty": lambda: fape_loss_uncrty(
                        out,
                        batch,
                        getattr(self.config, "fape_uncrty"),
                    ),
                    "supervised_chi_uncrty": lambda: supervised_chi_loss(
                        out["angles_uncrty"],
                        out["unnormalized_angles_uncrty"],
                        **{**batch, **getattr(self.config, "supervised_chi_uncrty")},
                    ),
                    "violation_uncrty": lambda: violation_loss(
                        out["violation_uncrty"],
                        **batch,
                    ),
                    "rotation_mse_uncrty": lambda: rotation_mse_loss_uncrty(
                        out,
                        batch,
                    ),
                    "bb_update_uncrty": lambda: bb_update_loss(out["bb_update_weight_uncrty"]),
                }
            )

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            if loss_name in self.config.keys():
                weight = self.config[loss_name].weight  # type: ignore
                loss = loss_fn()
                loss = torch.mean(loss)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"\n{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0.0, requires_grad=True)
                cum_loss = cum_loss + weight * loss
                losses[loss_name] = loss.detach().clone()

        # losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # seq_len = torch.mean(batch["seq_length"].float())
        # crop_len = batch["aatype"].shape[-1]
        # cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()  # type: ignore

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses


class P2DFlowLoss(nn.Module):
    """Aggregation of the various losses used in P2DFlow"""

    def __init__(self, custom_config: DictConfig):
        super().__init__()
        assert (
            "deterministic" in custom_config
        ), f"Please specify `deterministic` in custom_config: {custom_config} for loss computation."
        self.use_uncertainty_loss = not custom_config.pop("deterministic")

        self.config = config.loss

        self.config = merge_config_dict(self.config, OmegaConf.to_container(custom_config))  # type: ignore

    def forward(self, out, batch, _return_breakdown=True):
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["positions"][-1],
                **getattr(self.config, "violation"),
            )
        if "violation_uncrty" not in out.keys() and self.use_uncertainty_loss:
            out["violation_uncrty"] = find_structural_violations(
                batch,
                out["positions_uncrty"][-1],
                **getattr(self.config, "violation"),
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["positions"][-1],
                )
            )

        loss_fns = {
            "neighbor_ca_dist": lambda: neighbor_ca_dist_loss(
                out["positions"], batch["all_atom_positions"], batch["all_atom_mask"], std=batch.get("neighbor_ca_std")
            ),
            "pairwise_dist": lambda: pairwise_dist_loss(
                out["positions"],
                batch["all_atom_positions"],
                batch["all_atom_mask"],
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                getattr(self.config, "fape"),
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["angles"],
                out["unnormalized_angles"],
                **{**batch, **getattr(self.config, "supervised_chi")},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **batch,
            ),
            "protein_mse": lambda: protein_aa_mse_loss(
                out,
                batch,
            ),
        }

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            if loss_name in self.config.keys():
                weight = self.config[loss_name].weight  # type: ignore
                loss = loss_fn()
                loss = torch.mean(loss)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"\n{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0.0, requires_grad=True)
                cum_loss = cum_loss + weight * loss
                losses[loss_name] = loss.detach().clone()

        # losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # seq_len = torch.mean(batch["seq_length"].float())
        # crop_len = batch["aatype"].shape[-1]
        # cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()  # type: ignore

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
