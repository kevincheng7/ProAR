# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import random
import typing as T
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn

from openfold.np import residue_constants
from src.models.structure_module import StructureModule


@dataclass
class StructureModuleConfig:
    uncrty_downscaled_fac: float = 1.0
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_skip: int = 64
    c_dssp: int = 64
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.2
    droppath_rate: float = 0.1
    no_blocks: int = 4
    no_blocks_refine: int = 2
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    no_heads_transformer: int = 4
    no_layers_transformer: int = 2
    no_layers_egnn: int = 2
    no_heads_swin: tuple = (8, 8, 8, 8)
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class FoldingTrunkConfig:
    _name: str = "FoldingTrunkConfig"
    num_blocks: int = 12
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: T.Optional[int] = None

    structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used
    in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """

    if mask is None:
        return None
    assert len(mask.shape) == 2
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


def get_positional_embedding(indices, embedding_dim, max_len=2056):
    """
    Creates sine / cosine positional embeddings from a prespecified indices.
    Copied from https://github.com/lujiarui/Str2Str/blob/main/src/models/net/denoising_ipa.py#L13.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embedding_dim: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embedding_dim]
    """
    K = torch.arange(embedding_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))).to(
        indices.device
    )
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))).to(
        indices.device
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Copied from https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FoldingTrunk(nn.Module):
    def __init__(
        self, horizon, use_embedding=True, use_edge_transition=True, deterministic=False, is_forecast=False, **kwargs
    ):
        super().__init__()
        self.cfg = FoldingTrunkConfig(**kwargs)
        self.horizon = horizon
        self.use_embedding = use_embedding
        self.deterministic = deterministic
        self.is_forecast = is_forecast
        self.use_edge_transition = use_edge_transition
        assert self.cfg.max_recycles > 0

        c_s = self.cfg.sequence_state_dim
        c_z = self.cfg.pairwise_state_dim

        assert c_s % self.cfg.sequence_head_width == 0
        assert c_z % self.cfg.pairwise_head_width == 0
        if self.use_embedding:
            block = Embedding
            self.cfg.num_blocks = 1
            self.cfg.max_recycles = 1
        else:
            raise ValueError("Currently we do not support using the default FolddingTrunk as in ESMfold")
            # block = TriangularSelfAttentionBlock

        self.pairwise_positional_embedding = RelativePosition(self.cfg.position_bins, c_z)
        self.timestep_embedding = TimestepEmbedder(self.cfg.structure_module["c_s"])  # type: ignore
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedding.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.timestep_embedding.mlp[2].weight, std=0.02)  # type: ignore

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.cfg.sequence_head_width,
                    pairwise_head_width=self.cfg.pairwise_head_width,
                    dropout=self.cfg.dropout,  # type: ignore
                )
                for _ in range(self.cfg.num_blocks)
            ]
        )

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()

        if self.deterministic:
            self.cfg.structure_module["no_blocks_refine"] = 0  # type: ignore

        self.structure_module = StructureModule(
            self.horizon,
            self.deterministic,
            self.use_edge_transition,
            **self.cfg.structure_module,  # type: ignore
        )
        self.trunk2sm_s = nn.Linear(c_s, self.structure_module.c_s)
        self.trunk2sm_z = nn.Linear(c_z, self.structure_module.c_z)

        self.chunk_size = self.cfg.chunk_size

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def forward(
        self,
        seq_feats,
        pair_feats,
        rigids1,
        rigids2,
        dssp,
        step,
        true_aa,
        residx,
        mask,
        no_recycles: T.Optional[int] = None,
    ):
        """
        Inputs:
          seq_feats:     B x L x C            tensor of sequence features
          pair_feats:    B x L x L x C        tensor of pair features
          rigids1:       B x L                initial Rigid
          rigids2:       B x L                last Rigid
          dssp:          B x L                Dictionary of protein secondary structure (DSSP) assignments
          step:          B                    interpolation timestep
          true_aa:       B x L                residue type
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        device = seq_feats.device
        s_s_0 = seq_feats  # (bs, num_res, c_s)
        s_z_0 = pair_feats  # (bs, num_res, c_z)

        if no_recycles is None:
            no_recycles = self.cfg.max_recycles
        else:
            assert no_recycles >= 0, "Number of recycles must not be negative."
            no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

        def trunk_iter(s, z, residx, mask):
            """may cause OOM on A800 GPU"""
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        def embedding(s, residx, true_aa, step, rigids1, dssp, mask):
            """note that recycling for embedding layer is disabled"""

            self_condition_ca = rigids1.get_trans()

            for block in self.blocks:
                s, z = block(true_aa, residx, step, self_condition_ca, s, dssp)
            return s, z

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        no_recycles = random.randint(1, self.cfg.max_recycles) if self.training else no_recycles
        assert no_recycles > 0
        for recycle_idx in range(no_recycles):
            with ExitStack() if recycle_idx == no_recycles - 1 else torch.no_grad():
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())

                if self.use_embedding:
                    s_s, s_z = embedding(s_s_0 + recycle_s, residx, true_aa, step, rigids1, dssp, mask)
                else:
                    s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                # === Timestep embedding ===
                t_embed = self.timestep_embedding(step)  # (bs, c_s)

                # === Structure module ===
                s_s = self.trunk2sm_s(s_s)
                s_z = self.trunk2sm_z(s_z)

                structure = self.structure_module(
                    {"single": s_s, "pair": s_z},
                    true_aa,
                    rigids1,
                    rigids2,
                    t_embed,
                    dssp,
                    mask.float(),
                )

                recycle_s = s_s
                recycle_z = s_z
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = FoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        assert isinstance(structure, dict)  # type: ignore
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins


class Embedding(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        d_t=128,
        d_pos=128,
        d_type=128,
        d_esm=128,
        d_dssp=128,
        num_types=residue_constants.restype_num,
        padding_idx=0,
        embed_self_conditioning=True,
        num_bins=22,
        min_bin=1e-5,
        max_bin=20.0,
        **kwargs,
    ):
        super(Embedding, self).__init__()
        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.raw_esm_size = sequence_state_dim
        self.d_esm = d_esm

        self.embed_self_conditioning = embed_self_conditioning
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin

        d_node = 0
        d_edge = 0

        # ------------ Timestep embedding ------------
        d_node += d_t
        d_edge += d_t * 2
        self.timestep_embedder = partial(TimestepEmbedder.timestep_embedding, dim=d_t)

        # ------------ Positional index embedding ------------
        d_node += d_pos
        d_edge += d_pos
        self.position_embedder = partial(get_positional_embedding, embedding_dim=d_pos)

        # ------------Amino acid type embedding ------------
        d_node += d_type
        d_edge += d_type * 2
        self.type_embedder = nn.Embedding(
            num_embeddings=num_types + 1, embedding_dim=d_type, padding_idx=padding_idx  # [MASK] type
        )

        # ------------ ESM embedding ------------
        d_node += d_esm
        d_edge += d_esm * 2
        if self.raw_esm_size != d_esm:
            self.esm_downsampler = nn.Linear(self.raw_esm_size, d_esm)

        # ------------ DSSP embedding ------------
        d_node += d_dssp
        d_edge += d_dssp * 2
        self.dssp_embedder = nn.Embedding(num_embeddings=3 + 1, embedding_dim=d_dssp, padding_idx=padding_idx)

        # ------------ Self-conditioning distogram ------------
        if embed_self_conditioning:
            d_edge += num_bins

        self.node_embedder = nn.Sequential(
            nn.Linear(d_node, self.sequence_state_dim),
            nn.ReLU(),
            nn.Linear(self.sequence_state_dim, self.sequence_state_dim),
            nn.ReLU(),
            nn.Linear(self.sequence_state_dim, self.sequence_state_dim),
            nn.LayerNorm(self.sequence_state_dim),
        )
        self.edge_embedder = nn.Sequential(
            nn.Linear(d_edge, self.pairwise_state_dim),
            nn.ReLU(),
            nn.Linear(self.pairwise_state_dim, self.pairwise_state_dim),
            nn.ReLU(),
            nn.Linear(self.pairwise_state_dim, self.pairwise_state_dim),
            nn.LayerNorm(self.pairwise_state_dim),
        )

    def calc_distogram(self, pos, min_bin, max_bin, num_bins):
        dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
        lower = torch.linspace(min_bin, max_bin, num_bins, dtype=pos.dtype, device=pos.device)
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
        return dgram

    def _cross_concat(self, feats_1d, batch_size, seq_len):
        return torch.cat(
            [
                torch.tile(feats_1d[:, :, None, :], (1, 1, seq_len, 1)),
                torch.tile(feats_1d[:, None, :, :], (1, seq_len, 1, 1)),
            ],
            dim=-1,
        ).reshape([batch_size, seq_len**2, -1])

    def forward(self, types, seq_idx, t, self_conditioning_ca, esm_embed, dssp):
        """
        Embeds a set of inputs.

        Args:
            types: [..., N_res] Amino acid type for each residue.
            seq_idx: [..., N_res] Positional sequence index.
            t: Sampled timestep.
            self_conditioning_ca: [..., N_res, 3] Ca positions of self-conditioning input.
            esm_embed: [..., N_res, D_esm] ESM embedding for each residue.
            dssp: [..., N_res] DSSP secondary structure assignments.

        Returns:
            node_embed: [batch_size, N_res, D_node]
            edge_embed: [batch_size, N_res, N_res, D_edge]
        """
        node_feats = []
        pair_feats = []

        batch_size, num_res = seq_idx.shape

        # ------------ Timestep embedding ------------
        prot_t_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (1, num_res, 1))  # [batch_size, N_res, D_t]

        node_feats.append(prot_t_embed)  # [batch_size, N_res, D_t]

        pair_feats.append(self._cross_concat(node_feats[-1], batch_size, num_res))  # [batch_size, (N_res)**2, D_t*2]

        # ------------ Positional embedding ------------
        node_feats.append(self.position_embedder(seq_idx))
        # relative 2d positional embedding
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([batch_size, (num_res) ** 2])
        pair_feats.append(self.position_embedder(rel_seq_offset))

        # ------------ Amino acid type embedding ------------
        type_embed = self.type_embedder(types)  # [batch_size, N_res, D_type]
        node_feats.append(type_embed)  # [batch_size, N_res, D_type]
        pair_feats.append(
            self._cross_concat(node_feats[-1], batch_size, num_res)
        )  # [batch_size, (N_res)**2, D_type*2]

        # ------------ ESM embedding ------------
        if self.raw_esm_size != self.d_esm:
            esm_embed = self.esm_downsampler(esm_embed)
        node_feats.append(esm_embed)  # [batch_size, N_res, D_esm]
        pair_feats.append(self._cross_concat(esm_embed, batch_size, num_res))  # [batch_size, (N_res)**2, D_esm*2]

        # ------------ DSSP embedding ------------
        dssp_embed = self.dssp_embedder(dssp)
        node_feats.append(dssp_embed)
        pair_feats.append(self._cross_concat(node_feats[-1], batch_size, num_res))

        # ------------ Self-conditioning distogram ------------
        if self.embed_self_conditioning:

            sc_dgram = self.calc_distogram(
                self_conditioning_ca.to(esm_embed.dtype),
                self.min_bin,
                self.max_bin,
                self.num_bins,
            )  # [batch_size, N_res, N_res, N_bins]
            pair_feats.append(
                sc_dgram.reshape([batch_size, (num_res) ** 2, -1])
            )  # [batch_size, (N_res + N_atom)**2, N_bins]

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1))  # [batch_size, N_res, D_node]
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1))  # [batch_size, N_res**2, D_edge]
        edge_embed = edge_embed.reshape([batch_size, num_res, num_res, -1])

        return node_embed, edge_embed
