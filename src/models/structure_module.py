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
from typing import Optional, Sequence

import torch
import torch.nn as nn

from openfold.model.primitives import LayerNorm, Linear
from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold.utils.feats import frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import dict_multimap
from src.models.modules import layers
from src.models.modules.drop_path import DropPath
from src.models.modules.egnn import EGNN
from src.models.modules.ipa import InvariantPointAttention, InvariantPointAttention2Rigids
from src.models.modules.mvn import BackboneMVNGlobular


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class StructureModuleBlock2Rigids(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_skip,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        no_heads_transformer,
        no_layers_transformer,
        inf,
        epsilon,
        dropout_rate,
        droppath_rate,
        use_edge_transition,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_skip:
                skip connection channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            no_heads_transformer:
                Number of Transformer heads
            no_layers_transformer:
                Number of Transformer layers
            inf:
                Large number used for attention masking
            epsilon:
                Small number used in angle resnet normalization
            dropout_rate:
                Dropout rate used throughout the layer
            droppath_rate;
                Drop path used throughout the layer
            use_edge_transition:
                Whether to update pair representation
        """
        super(StructureModuleBlock2Rigids, self).__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_s, 6 * c_s, bias=True))

        self.layer_norm_pre_ipa = nn.LayerNorm(c_s, elementwise_affine=False, eps=1e-6)
        self.ipa = InvariantPointAttention2Rigids(
            c_s,
            c_z,
            c_ipa,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            inf,
            epsilon,
        )
        self.dropout_post_ipa = nn.Dropout(dropout_rate)
        self.layer_norm_post_ipa = nn.LayerNorm(c_s, eps=1e-6)

        self.skip = Linear(c_s, c_skip, init="final")
        _in_dim = c_s + c_skip
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=_in_dim,
            nhead=no_heads_transformer,
            dim_feedforward=_in_dim,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, no_layers_transformer)
        self.linear_post_transformer = Linear(_in_dim, c_s, init="final")

        self.layer_norm_pre_transition = nn.LayerNorm(c_s, elementwise_affine=False, eps=1e-6)
        self.node_transition = layers.StructureModuleNodeTransitionLayer(c_s)
        self.dropout_post_transition = nn.Dropout(dropout_rate)
        self.layer_norm_post_transition = nn.LayerNorm(c_s)

        self.use_edge_transition = use_edge_transition
        if self.use_edge_transition:
            self.edge_transition = layers.StructureModuleEdgeTransitionLayer(c_s, c_z, c_z, dilation=4)

        self.bb_update1 = layers.BackboneUpdate(c_s)
        self.bb_update2 = layers.BackboneUpdate(c_s)

        self.drop_path = DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()

    def forward(
        self,
        s,
        s_initial,
        z,
        rigids1: Rigid,
        rigids2: Rigid,
        t_embed,
        mask,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            s_initial:
                [*, N_res, C_s] initial single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            rigids1:
                [*, N_res] initial Rigid
            rigids2:
                [*, N_res] last Rigid
            t_embed:
                [*, C_s] timestep embedding
            mask:
                [*, N_res] sequence mask
        """
        edge_mask = mask[..., None] * mask[..., None, :]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_embed).chunk(6, dim=1)
        s = s + self.drop_path(
            gate_msa.unsqueeze(1)
            * mask.unsqueeze(-1)
            * self.ipa(
                modulate(self.layer_norm_pre_ipa(s), shift_msa, scale_msa),
                z,
                rigids1,
                rigids2,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
        )
        s = self.dropout_post_ipa(s)
        s = self.layer_norm_post_ipa(s)

        s_cat = torch.cat([s, self.skip(s_initial)], dim=-1)
        s_cat = self.transformer(s_cat.transpose(0, 1), src_key_padding_mask=mask == 0).transpose(0, 1)
        s = s + self.linear_post_transformer(s_cat)

        s = s + self.drop_path(
            gate_mlp.unsqueeze(1)
            * self.node_transition(modulate(self.layer_norm_pre_transition(s), shift_mlp, scale_mlp))
        )
        s = self.dropout_post_transition(s)
        s = self.layer_norm_post_transition(s) * mask.unsqueeze(-1)

        rigids1 = rigids1.compose_q_update_vec(self.bb_update1(s))
        rigids2 = rigids2.compose_q_update_vec(self.bb_update2(s))

        if self.use_edge_transition:
            z = self.edge_transition(s, z) * edge_mask.unsqueeze(-1)

        return s, z, rigids1, rigids2


class StructureModuleBlock(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_skip,
        c_dssp,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        no_heads_transformer,
        no_layers_transformer,
        inf,
        epsilon,
        dropout_rate,
        use_edge_transition,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_skip:
                skip connection channel dimension
            c_dssp:
                DSSP secondary structure assignments channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            no_heads_transformer:
                Number of Transformer heads
            no_layers_transformer:
                Number of Transformer layers
            inf:
                Large number used for attention masking
            epsilon:
                Small number used in angle resnet normalization
            dropout_rate:
                Dropout rate used throughout the layer
            use_edge_transition:
                Whether to update pair representation
        """
        super(StructureModuleBlock, self).__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_s, 6 * c_s, bias=True))

        self.layer_norm_pre_ipa = nn.LayerNorm(c_s, elementwise_affine=False, eps=1e-6)
        self.ipa = InvariantPointAttention(
            c_s,
            c_z,
            c_ipa,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            inf,
            epsilon,
        )
        self.dropout_post_ipa = nn.Dropout(dropout_rate)
        self.layer_norm_post_ipa = nn.LayerNorm(c_s, eps=1e-6)

        self.skip = Linear(c_s, c_skip, init="final")
        self.skip_dssp = Linear(c_dssp, c_skip, init="final")
        _in_dim = c_s + c_skip + c_skip
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=_in_dim,
            nhead=no_heads_transformer,
            dim_feedforward=_in_dim,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, no_layers_transformer)
        self.linear_post_transformer = Linear(_in_dim, c_s, init="final")

        self.layer_norm_pre_transition = nn.LayerNorm(c_s, elementwise_affine=False, eps=1e-6)
        self.node_transition = layers.StructureModuleNodeTransitionLayer(c_s)
        self.dropout_post_transition = nn.Dropout(dropout_rate)
        self.layer_norm_post_transition = nn.LayerNorm(c_s)

        self.use_edge_transition = use_edge_transition
        if self.use_edge_transition:
            self.edge_transition = layers.StructureModuleEdgeTransitionLayer(c_s, c_z, c_z, dilation=4)

        self.bb_update = layers.BackboneUpdate(c_s)

    def forward(
        self,
        s,
        s_initial,
        z,
        rigids: Rigid,
        t_embed,
        dssp,
        mask,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            s_initial:
                [*, N_res, C_s] initial single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            rigids:
                [*, N_res] Rigid
            t_embed:
                [*, C_s] timestep embedding
            dssp:
                [*, N_res] DSSP secondary structure assignments
            mask:
                [*, N_res] sequence mask
        """
        edge_mask = mask[..., None] * mask[..., None, :]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_embed).chunk(6, dim=1)
        s = s + gate_msa.unsqueeze(1) * mask.unsqueeze(-1) * self.ipa(
            modulate(self.layer_norm_pre_ipa(s), shift_msa, scale_msa),
            z,
            rigids,
            mask,
            inplace_safe=inplace_safe,
            _offload_inference=_offload_inference,
            _z_reference_list=z_reference_list,
        )
        s = self.dropout_post_ipa(s)
        s = self.layer_norm_post_ipa(s)

        s_cat = torch.cat([s, self.skip(s_initial), self.skip_dssp(dssp)], dim=-1)
        s_cat = self.transformer(s_cat.transpose(0, 1), src_key_padding_mask=mask == 0).transpose(0, 1)
        s = s + self.linear_post_transformer(s_cat)

        s = s + gate_mlp.unsqueeze(1) * self.node_transition(
            modulate(self.layer_norm_pre_transition(s), shift_mlp, scale_mlp)
        )
        s = self.dropout_post_transition(s)
        s = self.layer_norm_post_transition(s) * mask.unsqueeze(-1)

        rigids = rigids.compose_q_update_vec(self.bb_update(s))

        if self.use_edge_transition:
            z = self.edge_transition(s, z) * edge_mask.unsqueeze(-1)

        return s, z, rigids


class StructureModule(nn.Module):
    def __init__(
        self,
        horizon,
        deterministic,
        use_edge_transition,
        uncrty_downscaled_fac,
        c_s,
        c_z,
        c_ipa,
        c_skip,
        c_dssp,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        droppath_rate,
        no_blocks,
        no_blocks_refine,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        no_heads_transformer,
        no_layers_transformer,
        no_layers_egnn,
        trans_scale_factor,
        epsilon,
        inf,
        sigma_translation: float = 1.0,
        covariance_model: str = "globular",
        complex_scaling: bool = False,
        **kwargs,
    ):
        """
        Args:
            horizon:
                Horizon of the network
            deterministic:
                If False, use an uncertainty prediction module to fit the covariance
            use_edge_transition:
                Whether to update pair representation
            uncrty_downscaled_fac:
                Uncertainty downscaling factor used in model inference, should between [0.0, 1.0].
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_skip:
                skip connection channel dimension
            c_dssp:
                DSSP secondary structure assignments channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            no_heads_transformer:
                Number of Transformer heads
            no_layers_transformer:
                Number of Transformer layers
            no_layers_egnn:
                Number of EGNN layers
            dropout_rate:
                Dropout rate used throughout the layer
            droppath_rate;
                Drop path used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_blocks_refine:
                Number of structure module blocks for refinement.
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            sigma_translation (float, optional):
                Scaling factor for the translation component of the covariance matrix. Defaults to 1.0.
            covariance_model (str, optional):
                covariance mode,. Defaults to "globular".
            complex_scaling (bool, optional):
                Whether to scale the complex component of the covariance matrix by the translation component. Defaults to False.
        """
        super(StructureModule, self).__init__()

        self.horizon = horizon
        self.deterministic = deterministic
        self.use_edge_transition = use_edge_transition
        self.uncrty_downscaled_fac = uncrty_downscaled_fac
        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_skip = c_skip
        self.c_dssp = c_dssp
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.no_blocks = no_blocks
        self.no_blocks_refine = no_blocks_refine
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.no_heads_transformer = no_heads_transformer
        self.no_layers_transformer = no_layers_transformer
        self.no_layers_egnn = no_layers_egnn
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)
        self.dssp_embedder = nn.Embedding(3 + 1, self.c_dssp, padding_idx=0)

        self.dpr = [x.item() for x in torch.linspace(0, self.droppath_rate, self.no_blocks)]
        self.structure_module_blocks_2rigids = nn.ModuleList(
            [
                StructureModuleBlock2Rigids(
                    self.c_s,
                    self.c_z,
                    self.c_ipa,
                    self.c_skip,
                    self.no_heads_ipa,
                    self.no_qk_points,
                    self.no_v_points,
                    self.no_heads_transformer,
                    self.no_layers_transformer,
                    self.inf,
                    self.epsilon,
                    self.dropout_rate,
                    self.dpr[i],
                    self.use_edge_transition,
                )
                for i in range(self.no_blocks)
            ]
        )
        for block in self.structure_module_blocks_2rigids:
            # Zero-out adaLN modulation layers in structure module blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

        self.egnn_blocks = nn.ModuleList(
            [
                EGNN(
                    self.no_layers_egnn,
                    self.c_s,
                    0,
                )
                for _ in range(self.no_blocks)
            ]
        )

        self.angle_resnet = layers.AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

        self.average_rigids = layers.WeightedAverageRigids(self.c_s)

        # Zero-out adaLN modulation layers in average layer and uncertainty layer:
        nn.init.constant_(self.average_rigids.adaLN_modulation[-1].weight, 0)  # type: ignore
        nn.init.constant_(self.average_rigids.adaLN_modulation[-1].bias, 0)  # type: ignore

        if not self.deterministic:
            self.structure_module_blocks = nn.ModuleList(
                [
                    StructureModuleBlock(
                        self.c_s,
                        self.c_z,
                        self.c_ipa,
                        self.c_skip,
                        self.c_dssp,
                        self.no_heads_ipa,
                        self.no_qk_points,
                        self.no_v_points,
                        self.no_heads_transformer,
                        self.no_layers_transformer,
                        self.inf,
                        self.epsilon,
                        self.dropout_rate,
                        use_edge_transition=False,
                    )
                    for _ in range(self.no_blocks_refine)
                ]
            )
            for block in self.structure_module_blocks:
                # Zero-out adaLN modulation layers in structure module blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

            self.egnn_blocks_refine = nn.ModuleList(
                [
                    EGNN(
                        self.no_layers_egnn,
                        self.c_s,
                        0,
                    )
                    for _ in range(self.no_blocks_refine)
                ]
            )

            self.uncertainty_net = layers.UncertaintyPrediction(
                self.c_s, self.c_dssp, self.c_resnet, self.no_resnet_blocks
            )

            if covariance_model == "globular":
                self.base_gaussian = BackboneMVNGlobular(
                    sigma_translation=sigma_translation,
                    covariance_model=covariance_model,
                    complex_scaling=complex_scaling,
                )
            else:
                raise ValueError(
                    "Only Rg-confined Globular Polymer Covariance Model is supported."
                    "Please set covariance_model = 'globular'."
                )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        rigids1: Rigid,
        rigids2: Rigid,
        t_embed,
        dssp,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
               Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid and ligand atom indices
            rigids1:
                [*, N_res] initial Rigid
            rigids2:
                [*, N_res] last Rigid
            t_embed:
                [*, C_s] timestep embedding
            dssp:
                [*, N_res] DSSP secondary structure assignments
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        rigids1 = rigids1.scale_translation(1 / self.trans_scale_factor)
        rigids2 = rigids2.scale_translation(1 / self.trans_scale_factor)

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_dssp]
        dssp = self.dssp_embedder(dssp)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        outputs = []
        for i, (block, egnn) in enumerate(zip(self.structure_module_blocks_2rigids, self.egnn_blocks)):
            if i % 2 == 0:
                x, s = egnn(s, rigids1.get_trans())
                rigids1 = Rigid(rigids1.get_rots(), x)
            else:
                x, s = egnn(s, rigids2.get_trans())
                rigids2 = Rigid(rigids2.get_rots(), x)
            # [*, N, C_s]
            s, z, rigids1, rigids2 = block(
                s,
                s_initial,
                z,
                rigids1,
                rigids2,
                t_embed,
                mask,
                inplace_safe,
                _offload_inference,
                z_reference_list,
            )
            rigids = self.average_rigids(s, t_embed, rigids1, rigids2)

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # [*, N_res, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            protein_pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": protein_pred_xyz,
                "states": s,
                "bb_update_weight": (block.bb_update1.linear.weight + block.bb_update2.linear.weight),  # type: ignore
            }

            outputs.append(preds)

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        if not self.deterministic:
            outputs.update(
                self.uncertainty_prediction_chroma(
                    s,
                    s_initial,
                    z,
                    backb_to_global,
                    t_embed,
                    dssp,
                    aatype,
                    mask,
                )
            )

        del z, z_reference_list

        return outputs

    def uncertainty_prediction_chroma(
        self,
        s,
        s_initial,
        z,
        backb_to_global: Rigid,
        t_embed,
        dssp,
        aatype,
        mask,
    ):
        bs, n = s.shape[:2]
        s = s.detach()
        s_initial = s_initial.detach()
        z = z.detach()

        len_p = mask.sum(-1).to(torch.long)

        # covariance prediction
        L = self.uncertainty_net(s, s_initial, dssp)  # (bs, n_res, n_res) lower triangular matrix
        if not self.training:
            L = L * self.uncrty_downscaled_fac

        # protein backbone unncertainty
        gZ_p = torch.zeros(bs, n, 1, 3, device=s.device)
        scale_tril_p = torch.zeros(bs, n, n, device=s.device)

        for b, l in zip(range(bs), len_p):
            dist = torch.distributions.MultivariateNormal(
                torch.zeros(l, device=s.device), scale_tril=L[b, :l, :l].float()
            )

            # (3, l) ==> (1, l, 1, 3)
            Z = dist.rsample(torch.Size([3])).permute([1, 0]).unsqueeze(0).unsqueeze(-2)
            C = torch.ones(1, l, device=s.device)
            gZ = self.base_gaussian._multiply_R_wo_R_center(Z, C, expand_heavy_atom=False)  # (1, l, 1, 3)

            R, RRt = self.base_gaussian._materialize_RRt_wo_R_center(C, expand_heavy_atom=False)  # (l, l)
            S = L[b, :l, :l]  # (l, l)
            scale_tril = R @ S  # (l, l)

            gZ_p[b, :l] = gZ.squeeze(0)
            scale_tril_p[b, :l, :l] = scale_tril

        trans_update = gZ_p.squeeze(-2)

        rigids_uncrty = Rigid(
            Rotation(rot_mats=backb_to_global.get_rots().get_rot_mats().detach(), quats=None),
            backb_to_global.get_trans().detach() + trans_update,
        )
        rigids_uncrty = rigids_uncrty.scale_translation(1 / self.trans_scale_factor)

        outputs = []
        for block, egnn in zip(self.structure_module_blocks, self.egnn_blocks_refine):
            x, s = egnn(s, rigids_uncrty.get_trans())
            rigids_uncrty = Rigid(rigids_uncrty.get_rots(), x)
            s, z, rigids_uncrty = block(s, s_initial, z, rigids_uncrty, t_embed, dssp, mask)
            backb_to_global_uncrty = rigids_uncrty.scale_translation(self.trans_scale_factor)

            # [*, N_res, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global_uncrty = self.torsion_angles_to_frames(
                backb_to_global_uncrty,
                angles,
                aatype,
            )

            protein_pred_xyz_uncrty = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global_uncrty,
                aatype,
            )

            preds_uncrty = {
                "frames_uncrty": backb_to_global_uncrty.to_tensor_7(),
                "sidechain_frames_uncrty": all_frames_to_global_uncrty.to_tensor_4x4(),
                "unnormalized_angles_uncrty": unnormalized_angles,
                "angles_uncrty": angles,
                "positions_uncrty": protein_pred_xyz_uncrty,
                "bb_update_weight_uncrty": block.bb_update.linear.weight,  # type: ignore
            }

            outputs.append(preds_uncrty)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["scale_tril_p"] = scale_tril_p
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)  # type: ignore

    def frames_and_literature_positions_to_atom14_pos(self, r, f):  # [*, N, 8]  # [*, N]
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
