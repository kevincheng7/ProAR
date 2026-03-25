"""ref: https://github.com/YuzheWangPKU/DiffPepBuilder/blob/main/analysis/metrics.py"""

import logging
import numpy as np
import torch
from tmtools import tm_align
from torchmetrics import Metric

log = logging.getLogger(__name__)

from openfold.np import residue_constants
from src.utilities.utils import tensor_to_ndarray


ca_ca = residue_constants.ca_ca
c_radius = residue_constants.van_der_waals_radius["C"]
restypes_with_x = list("ARNDCQEGHILKMFPSTWYV") + ["X"]
aatype_to_seq = lambda aatype: "".join([restypes_with_x[x] for x in aatype])


class BaseProteinMetric(Metric):
    """Base class for all protein metrics"""

    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_element", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute(self):
        return self.value / self.num_element  # type: ignore


class CaBondDeviation(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords=None, aatype=None):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)

            default tolerance value is adopted from https://github.com/bytedance/ConfDiff/blob/main/src/analysis/struct_quality.py#L307
        """
        device = pred_p_coords.device

        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for pred_ca in pred_coords_masked_ca:
            ca_ca_bond_dev, _ = self.ca_ca_distance(tensor_to_ndarray(pred_ca))
            self.value += torch.tensor(ca_ca_bond_dev, device=device)
            self.num_element += torch.tensor(1, device=device)

    def ca_ca_distance(self, ca_pos, tol=0.4):
        ca_bond_dists = np.linalg.norm(ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
        ca_ca_dev = np.mean(np.abs(ca_bond_dists - ca_ca))
        ca_ca_disconnect = np.mean(ca_bond_dists > (ca_ca + tol))
        return ca_ca_dev, ca_ca_disconnect


class CaDisconnectRate(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords=None, aatype=None):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)

            default tolerance value is adopted from https://github.com/bytedance/ConfDiff/blob/main/src/analysis/struct_quality.py#L307
        """
        device = pred_p_coords.device

        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for pred_ca in pred_coords_masked_ca:
            _, ca_ca_valid_percent = self.ca_ca_distance(tensor_to_ndarray(pred_ca))
            self.value += torch.tensor(ca_ca_valid_percent, device=device)
            self.num_element += torch.tensor(1, device=device)

    def ca_ca_distance(self, ca_pos, tol=0.4):
        ca_bond_dists = np.linalg.norm(ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
        ca_ca_dev = np.mean(np.abs(ca_bond_dists - ca_ca))
        ca_ca_disconnect = np.mean(ca_bond_dists > (ca_ca + tol))
        return ca_ca_dev, ca_ca_disconnect


class NumCaStericClashes(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords=None, aatype=None):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)

            default tolerance value is adopted from https://github.com/bytedance/ConfDiff/blob/main/src/analysis/struct_quality.py#L284
        """
        device = pred_p_coords.device

        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for pred_ca in pred_coords_masked_ca:
            num_ca_steric_clashes, _ = self.ca_ca_clashes(tensor_to_ndarray(pred_ca))
            self.value += torch.tensor(num_ca_steric_clashes, dtype=torch.float32, device=device)
            self.num_element += torch.tensor(1, device=device)

    def ca_ca_clashes(self, ca_pos, tol=0.4):
        ca_ca_dists2d = np.linalg.norm(ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
        inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
        clashes = inter_dists < 2 * c_radius - tol
        return np.sum(clashes), np.mean(clashes)


class CaStericClashRate(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords=None, aatype=None):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)

            default tolerance value is adopted from https://github.com/bytedance/ConfDiff/blob/main/src/analysis/struct_quality.py#L284
        """
        device = pred_p_coords.device

        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for pred_ca in pred_coords_masked_ca:
            _, ca_steric_clash_percent = self.ca_ca_clashes(tensor_to_ndarray(pred_ca))
            self.value += torch.tensor(ca_steric_clash_percent, device=device)
            self.num_element += torch.tensor(1, device=device)

    def ca_ca_clashes(self, ca_pos, tol=0.4):
        ca_ca_dists2d = np.linalg.norm(ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
        inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
        clashes = inter_dists < 2 * c_radius - tol
        return np.sum(clashes), np.mean(clashes)


class PeptideTmScore(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords, aatype):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)
        """
        device = pred_p_coords.device

        gt_coords_masked = gt_p_coords * p_mask[..., None]
        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for gt_ca, pred_ca, aa in zip(gt_coords_masked_ca, pred_coords_masked_ca, aatype):
            seq = aatype_to_seq(aa)
            _, tm_score = self.calc_tm_score(tensor_to_ndarray(pred_ca), tensor_to_ndarray(gt_ca), seq, seq)

            self.value += torch.tensor(tm_score, device=device)
            self.num_element += torch.tensor(1, device=device)

    def calc_tm_score(self, pos_1, pos_2, seq_1, seq_2):
        tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
        return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


class PeptideRMSD(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords, aatype):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)
        """
        device = pred_p_coords.device

        gt_coords_masked = gt_p_coords * p_mask[..., None]
        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for gt_ca, pred_ca in zip(gt_coords_masked_ca, pred_coords_masked_ca):
            peptide_rmsd, flip = self.calc_rmsd(tensor_to_ndarray(pred_ca), tensor_to_ndarray(gt_ca), flip_align=True)

            self.value += torch.tensor(peptide_rmsd, device=device)
            self.num_element += torch.tensor(1, device=device)

    def calc_rmsd(self, pos_1, pos_2, flip_align=True):
        """
        Calculate RMSD. Consider both head-to-head and head-to-tail alignments and returns the minimum RMSD if flip_align is True.

        Returns:
            rmsd: Calculated minimum RMSD value.
            flip: Boolean flag indicating whether the head-to-tail alignment was used.
        """
        rmsd = np.mean(np.linalg.norm(pos_1 - pos_2, axis=-1))
        pos_1_reverse = np.flip(pos_1, axis=0)
        rmsd_reverse = np.mean(np.linalg.norm(pos_1_reverse - pos_2, axis=-1))
        if flip_align:
            flip = rmsd_reverse < rmsd
            rmsd = rmsd_reverse if flip else rmsd
        else:
            flip = False
        return rmsd, flip


class PeptideAlignedRMSD(BaseProteinMetric):
    def update(self, pred_p_coords, p_mask, gt_p_coords, aatype):
        """
        Args:
            pred_p_coords, gt_p_coords (Tensor): protein coordinates of shape (bs, n_res, 37, 3)
            p_mask (Tensor): protein all atom mask of shape (bs, n_res, 37)
            aatype (Tensor): protein aatype of shape (bs, num_res)
        """
        device = pred_p_coords.device

        gt_coords_masked = gt_p_coords * p_mask[..., None]
        pred_coords_masked = pred_p_coords * p_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = p_mask[..., ca_pos]

        for gt_ca, pred_ca in zip(gt_coords_masked_ca, pred_coords_masked_ca):
            peptide_aligned_rmsd = self.calc_aligned_rmsd(tensor_to_ndarray(pred_ca), tensor_to_ndarray(gt_ca))

            self.value += torch.tensor(peptide_aligned_rmsd, device=device)
            self.num_element += torch.tensor(1, device=device)

    def calc_aligned_rmsd(self, pos_1, pos_2):
        aligned_pos_1 = self.rigid_transform_3D(pos_1, pos_2)[0]
        return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

    def rigid_transform_3D(self, A, B, verbose=False):
        # Transforms A to look like B
        # https://github.com/nghiaho12/rigid_transform_3D
        assert A.shape == B.shape
        A = A.T
        B = B.T

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # sanity check
        # if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        reflection_detected = False
        if np.linalg.det(R) < 0:
            if verbose:
                log.debug("det(R) < R, reflection detected, correcting for it ...")
            Vt[2, :] *= -1
            R = Vt.T @ U.T
            reflection_detected = True

        t = -R @ centroid_A + centroid_B
        optimal_A = R @ A + t

        return optimal_A.T, R, t, reflection_detected
