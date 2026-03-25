from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from openfold.data.data_transforms import make_atom14_masks
from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold.utils.feats import (
    atom14_to_atom37,
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import dict_multimap
from src.models._base_model import BaseModel
from src.models.modules.egnn import EGNN
from src.models.modules.ipa import InvariantPointAttention
from src.models.modules.layers import BackboneUpdate, EdgeEmbedder, EnergyAdapter, NodeEmbedder


NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


class ModifiedP2DFlowForecast(BaseModel):

    def __init__(self, model_conf, with_time_emb=True, **kwargs):
        super().__init__(**kwargs)

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        self.energy_adapter = EnergyAdapter(
            d_node=model_conf.node_embed_size, n_head=model_conf.ipa.no_heads, p_drop=model_conf.dropout
        )

        self.num_torsions = 7
        self.torsions_pred_layer1 = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        self.torsions_pred_layer2 = nn.Linear(self._ipa_conf.c_s, self.num_torsions * 2)

        self.trunk = nn.ModuleDict()
        for b in range(self._model_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(**self._ipa_conf, narrow_z=True)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)

            self.trunk[f"energy_adapter_{b}"] = EnergyAdapter(
                d_node=model_conf.node_embed_size, n_head=model_conf.ipa.no_heads, p_drop=model_conf.dropout
            )

            self.trunk[f"egnn_{b}"] = EGNN(num_layers=2, hidden_dim=model_conf.node_embed_size, edge_feat_dim=0)

        self.bb_update_layer = BackboneUpdate(self._ipa_conf.c_s)

    def forward(self, inputs: Dict, time: Tensor, condition: Sequence[Dict]):
        """
        Args:
            inputs (dict): x_t, the perturbed interpolated data
            time (tensor): time infomation
            condition ((dict, dict)): contains x_0 and the static condition,
                i.e., the initial conditions at time t0 and the esm, esm_pair, energy infomation.
        """
        initial_condition, static_condition = condition

        node_mask = inputs["backbone_rigid_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        aatype = inputs["aatype"]

        node_repr_pre = static_condition["esm"]
        pair_repr_pre = static_condition["esm_pair"]
        energy = static_condition["energy"]

        rigids_t = Rigid.from_tensor_4x4(inputs["backbone_rigid_tensor"])
        rigids_sc = Rigid.from_tensor_4x4(initial_condition["backbone_rigid_tensor"])

        trans_t = rigids_t.get_trans()
        trans_sc = rigids_sc.get_trans()

        init_node_embed = self.node_embedder(time.unsqueeze(-1), aatype, node_repr_pre, node_mask)
        init_node_embed = init_node_embed * node_mask[..., None]

        init_node_embed = self.energy_adapter(init_node_embed, energy, mask=node_mask)
        init_node_embed = init_node_embed * node_mask[..., None]

        init_edge_embed = self.edge_embedder(init_node_embed, trans_t, trans_sc, pair_repr_pre, edge_mask)
        init_edge_embed = init_edge_embed * edge_mask[..., None]

        curr_rigids = self.rigids_ang_to_nm(rigids_t)

        node_embed = init_node_embed
        edge_embed = init_edge_embed

        outputs = []
        for b in range(self._model_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, node_mask)  # (B,L,d_node)
            ipa_embed = node_embed + ipa_embed

            ipa_embed = ipa_embed * node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](ipa_embed)

            node_embed = self.trunk[f"energy_adapter_{b}"](node_embed, energy, mask=node_mask)
            node_embed = node_embed * node_mask[..., None]

            _, node_embed = self.trunk[f"egnn_{b}"](node_embed, trans_t)
            node_embed = node_embed * node_mask[..., None]

            if b == (self._model_conf.num_blocks - 1):
                curr_rigids = curr_rigids.compose_q_update_vec(self.bb_update_layer(node_embed))

                backb_to_global = Rigid(
                    Rotation(rot_mats=curr_rigids.get_rots().get_rot_mats(), quats=None),
                    curr_rigids.get_trans(),
                )
                backb_to_global = self.rigids_nm_to_ang(backb_to_global)

                unnormalized_angles = node_embed + self.torsions_pred_layer1(node_embed)
                unnormalized_angles = self.torsions_pred_layer2(unnormalized_angles).reshape(
                    aatype.shape + (self.num_torsions, 2)
                )  # (B,L,self.num_torsions,2)
                norm_denom = torch.sqrt(
                    torch.sum(unnormalized_angles**2, dim=-1, keepdim=True)
                )  # (B,L,self.num_torsions,1)
                angles = unnormalized_angles / norm_denom  # (B,L,self.num_torsions,2)

                all_frames_to_global = self.torsion_angles_to_frames(
                    backb_to_global,
                    angles,
                    aatype,
                )

                protein_pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype,
                )

                scaled_rigids = self.rigids_nm_to_ang(curr_rigids)

                preds = {
                    "frames": scaled_rigids.to_tensor_7(),
                    "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                    "unnormalized_angles": unnormalized_angles,
                    "angles": angles,
                    "positions": protein_pred_xyz,
                    "states": node_embed,
                }
                outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = node_embed

        outputs["aatype"] = aatype
        make_atom14_masks(outputs)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            outputs[k] *= node_mask.unsqueeze(-1)

        outputs["final_atom_positions"] = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs["final_atom_mask"] = outputs["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["frames"][-1]

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

