# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from openfold.model.primitives import Linear
from openfold.utils.rigid_utils import Rigid, Rotation
from src.utilities.utils import build_covariance_from_scaling_rotation, build_scaling_rotation


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rbf(D, D_min=0.0, D_max=20.0, D_count=36):
    # Distance radial basis function
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None, :]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))  # (B, L, L, D_count)
    return RBF


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """ref: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""

    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(
        indices.device
    )
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(
        indices.device
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


def add_RoPE(indices):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: (B,L,embed_size)
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [B, L, embed_size]
    """
    seq_len, embed_size = indices.shape[-2:]
    seq_all = torch.arange(seq_len, device=indices.device)[:, None]  # (L,1)
    theta_all = torch.pow(1e4, torch.arange(embed_size) // 2 / -embed_size)[None, :]  # (1,embed_size)
    sinusoidal_pos = (seq_all * theta_all.to(indices.device))[None, ...]  # (1,L,embed_size)

    cos_pos = torch.cos(sinusoidal_pos)  # (1,L,embed_size)
    sin_pos = torch.sin(sinusoidal_pos)  # (1,L,embed_size)
    indices_sin = torch.stack([-indices[..., 1::2], indices[..., ::2]], dim=-1)  # (B,L,embed_size/2,2)
    indices_sin = indices_sin.reshape(indices.shape)  # (B,L,embed_size)
    indices = indices * cos_pos + indices_sin * sin_pos
    return indices


class NodeEmbedder(nn.Module):
    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.c_node_pre = 1280
        self.aatype_emb_dim = self._cfg.c_pos_emb

        self.aatype_emb = nn.Embedding(21, self.aatype_emb_dim)

        total_node_feats = self.aatype_emb_dim + self._cfg.c_timestep_emb + self.c_node_pre

        self.linear = nn.Sequential(
            nn.Linear(total_node_feats, self.c_s),
            nn.ReLU(),
            nn.Dropout(self._cfg.dropout),
            nn.Linear(self.c_s, self.c_s),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.c_timestep_emb, max_positions=2056)[:, None, :].repeat(
            1, mask.shape[1], 1
        )
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, aatype, node_repr_pre, mask):

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        aatype_emb = self.aatype_emb(aatype) * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [aatype_emb]
        # timesteps are between 0 and 1. Convert to integers.
        time_emb = self.embed_t(timesteps, mask)
        input_feats.append(time_emb)

        input_feats.append(node_repr_pre)

        out = self.linear(torch.cat(input_feats, dim=-1))  # (B,L,d_node)

        return add_RoPE(out)


class EdgeEmbedder(nn.Module):
    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        self.num_cross_heads = 32
        self.c_pair_pre = 20
        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2 + self.c_pair_pre
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Dropout(self._cfg.dropout),
            nn.Linear(self.c_p, self.c_p),
        )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):

        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res, num_res, -1])
        )

    def forward(self, s, t, sc_t, pair_repr_pre, p_mask):

        num_batch, num_res, d_node = s.shape
        p_i = self.linear_s_p(s)  # (B,L,feat_dim)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        pos = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(pos)

        pos = t
        dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B,L,L)

        dist_feats = rbf(dists_2d, D_min=0.0, D_max=self._cfg.max_dist, D_count=self._cfg.num_bins)

        pos = sc_t
        dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B,L,L)

        sc_feats = rbf(dists_2d, D_min=0.0, D_max=self._cfg.max_dist, D_count=self._cfg.num_bins)

        all_edge_feats = torch.concat([cross_node_feats, relpos_feats, dist_feats, sc_feats, pair_repr_pre], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)  # (B,L,L,c_p)
        return edge_feats


class EnergyAdapter(nn.Module):
    def __init__(self, d_node=256, n_head=8, p_drop=0.1, ff_factor=1):
        super().__init__()

        self.d_node = d_node
        self.tfmr_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_node,
            nhead=n_head,
            dim_feedforward=ff_factor * d_node,
            dropout=p_drop,
            batch_first=True,
            norm_first=False,
        )

    def forward(self, node, energy, mask=None):
        energy_emb = get_time_embedding(energy, self.d_node, max_positions=2056)[:, None, :]  # (B,1,d_node)

        if mask is not None:
            mask = (1 - mask).bool()
        node = self.tfmr_layer(node, energy_emb, tgt_key_padding_mask=mask)

        return node


class WeightedAverageRigids(nn.Module):
    def __init__(self, c_s):
        super(WeightedAverageRigids, self).__init__()

        self.norm_final = nn.LayerNorm(c_s, elementwise_affine=False, eps=1e-6)
        self.linear_trans = nn.Linear(c_s, 1, bias=True)
        self.linear_quats = nn.Linear(c_s, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_s, 2 * c_s, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, t_embed, r1: Rigid, r2: Rigid) -> Rigid:
        shift, scale = self.adaLN_modulation(t_embed).chunk(2, dim=1)
        s = modulate(self.norm_final(s), shift, scale)
        weight_trans = self.sigmoid(self.linear_trans(s))  # (b, n, 1)

        r1_trans = r1.get_trans()  # (b, n, 3)
        r2_trans = r2.get_trans()  # (b, n, 3)
        r_trans = r1_trans * weight_trans + r2_trans * (1 - weight_trans)

        r1_quats = r1.get_rots().get_quats()  # (b, n, 4)
        r2_quats = r2.get_rots().get_quats()  # (b, n, 4)
        weight_quats = self.sigmoid(self.linear_quats(s))  # (b, n, 1)
        r_quats = r1_quats * weight_quats + r2_quats * (1 - weight_quats)

        tensor_7 = torch.cat([r_quats, r_trans], dim=-1)
        r = Rigid.from_tensor_7(tensor_7, normalize_quats=True)

        return r


class ResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(ResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleNodeTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleNodeTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleEdgeTransitionLayer(nn.Module):
    def __init__(self, c_s, c_z_in, c_z_out, num_layers=2, dilation=2):
        super(StructureModuleEdgeTransitionLayer, self).__init__()

        c_bias = c_s // dilation
        c_hidden = c_bias * 2 + c_z_in
        self.c_z_out = c_z_out
        self.initial = Linear(c_s, c_bias, init="relu")

        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(c_hidden, c_hidden, init="relu"))
            trunk_layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*trunk_layers)
        self.final = Linear(c_hidden, c_z_out, init="final")
        if self.c_z_out != 1:
            self.layer_norm = nn.LayerNorm(c_z_out)

    def forward(self, s, z):
        s = self.initial(s)
        b, n, _ = s.shape

        bias = torch.cat(
            [
                torch.tile(s[:, :, None, :], (1, 1, n, 1)),
                torch.tile(s[:, None, :, :], (1, n, 1, 1)),
            ],
            dim=-1,
        )
        z = torch.cat([z, bias], dim=-1).reshape(b * n**2, -1)
        z = self.final(self.trunk(z) + z)
        z = self.layer_norm(z) if self.c_z_out != 1 else z
        z = z.reshape(b, n, n, -1)
        return z


class UncertaintyPredictionAnisotropic(nn.Module):
    """A MLP is used to predict the diffusion coefficient, which can be represented
    as the covariance of a 3d Gaussian distribution and further decomposed into
    the predicted quaternion `q` and the 3d scaling vector `s`.
    """

    def __init__(self, c_in, c_hidden, no_blocks):
        super(UncertaintyPredictionAnisotropic, self).__init__()

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scaling_modifier = 0.01

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.linear_out = Linear(self.c_hidden, 6, init="final")  # scaling vector(3) and quaternion(3).

    def get_covariance(self, s, r, scaling_modifier=1.0):
        return self.covariance_activation(self.scaling_activation(s), scaling_modifier, r)

    def forward(self, s, s_initial):
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)
        scaling, quat = self.linear_out(s).split([3, 3], dim=-1)

        quat = Rotation(
            quats=torch.cat([torch.ones(*quat.shape[:-1], 1, dtype=quat.dtype, device=quat.device), quat], dim=-1),
            normalize_quats=True,
        ).get_quats()

        cov = self.get_covariance(scaling, quat, scaling_modifier=self.scaling_modifier)

        return cov


class UncertaintyPredictionChroma(nn.Module):
    def __init__(self, c_in, c_dssp, c_hidden, no_blocks):
        super(UncertaintyPredictionChroma, self).__init__()

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scale_rot_activation = build_scaling_rotation
        self.scaling_modifier = 0.01

        self.c_in = c_in
        self.c_dssp = c_dssp
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)
        self.linear_dssp = Linear(self.c_dssp, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.linear_out_protein = Linear(self.c_hidden, 1, init="final")  # scaling vector(1) for each CA.
        self.linear_out_ligand = Linear(
            self.c_hidden, 6, init="final"
        )  # scaling vector(3) + quaternion(3) for each ligand atom.

    def get_covariance(self, s, r, scaling_modifier=1.0):
        return self.covariance_activation(self.scaling_activation(s), scaling_modifier, r)

    def get_scale_rot(self, s, r, scaling_modifier=1.0):
        return self.scale_rot_activation(self.scaling_activation(s) * scaling_modifier, r)

    def forward(self, s, s_initial, dssp, num_aa):
        bs, n, _ = s.shape

        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s_dssp = self.relu(dssp)
        s_dssp = self.linear_dssp(s_dssp)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial + s_dssp

        for l in self.layers:
            s = l(s)

        s = self.relu(s)
        s_protein, s_ligand = s.split([num_aa, n - num_aa], dim=-2)

        scale_protein = self.linear_out_protein(s_protein)
        scale_ligand, quat_ligand = self.linear_out_ligand(s_ligand).split([3, 3], dim=-1)

        quat_ligand = Rotation(
            quats=torch.cat(
                [
                    torch.ones(*quat_ligand.shape[:-1], 1, dtype=quat_ligand.dtype, device=quat_ligand.device),
                    quat_ligand,
                ],
                dim=-1,
            ),
            normalize_quats=True,
        ).get_quats()

        scale_protein = self.scaling_modifier * self.scaling_activation(scale_protein)
        scale_rot_ligand = self.get_scale_rot(scale_ligand, quat_ligand, scaling_modifier=self.scaling_modifier)

        return scale_protein, scale_rot_ligand


class UncertaintyPrediction(nn.Module):
    def __init__(self, c_in, c_dssp, c_hidden, no_blocks):
        super(UncertaintyPrediction, self).__init__()

        self.diag_activation = nn.functional.softplus
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scale_rot_activation = build_scaling_rotation
        self.scaling_modifier = nn.Parameter(torch.tensor(0.01))

        self.c_in = c_in
        self.c_dssp = c_dssp
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)
        self.linear_dssp = Linear(self.c_dssp, self.c_hidden)
        self.linear_multipliers = Linear(self.c_hidden, 2, init="final")

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.linear_out = Linear(self.c_hidden, 5, init="final")  # diagonal (1) and off-diagonal (4) elements.

    def get_covariance(self, s, r, scaling_modifier=1.0):
        return self.covariance_activation(self.diag_activation(s), scaling_modifier, r)

    def get_scale_rot(self, s, r, scaling_modifier=1.0):
        return self.scale_rot_activation(self.diag_activation(s) * scaling_modifier, r)

    def forward(self, s, s_initial, dssp):
        bs, N, _ = s.shape

        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s_dssp = self.relu(dssp)
        s_dssp = self.linear_dssp(s_dssp)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial + s_dssp

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        corr_out = self.linear_out(s)  # (bs, n, 5)
        L = torch.zeros(bs, N, N, device=corr_out.device, dtype=corr_out.dtype)
        idx = torch.arange(N)
        L[:, idx, idx] = corr_out[..., 0]

        # (i >= 1, corr_matrix[i, i-1] += corr_out[i, 1])
        if N > 1:
            indices = torch.arange(1, N)
            L[:, indices, indices - 1] += corr_out[:, indices, 1]

        # (i >= 2, corr_matrix[i, i-2] += corr_out[i, 2])
        if N > 2:
            indices = torch.arange(2, N)
            L[:, indices, indices - 2] += corr_out[:, indices, 2]

        # (i <= N-2, corr_matrix[i+1, i] += corr_out[i, 3])
        if N > 1:
            indices = torch.arange(N - 1)
            L[:, indices + 1, indices] += corr_out[:, indices, 3]

        # (i <= N-3, corr_matrix[i+2, i] += corr_out[i, 4])
        if N > 2:
            indices = torch.arange(N - 2)
            L[:, indices + 2, indices] += corr_out[:, indices, 4]

        a, b = self.linear_multipliers(s).split(1, dim=-1)

        mask = torch.eye(N, device=s.device)  # (n, n)
        L = torch.where(
            mask.bool(),
            self.diag_activation(L) * self.diag_activation(torch.diag_embed(a.squeeze(-1)))
            + self.diag_activation(torch.diag_embed(b.squeeze(-1))),
            L,
        )
        L = torch.tril(self.scaling_modifier * L)  # global scaling modifier

        return L
