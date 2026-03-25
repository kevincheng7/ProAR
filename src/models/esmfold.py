from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import esm
import torch
from esm import Alphabet
from torch import Tensor, nn

from openfold.config import config
from openfold.data.data_transforms import make_atom14_masks
from openfold.model.heads import DistogramHead, PerResidueLDDTCaPredictor
from openfold.np import residue_constants
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.loss import compute_plddt
from openfold.utils.rigid_utils import Rigid
from src.models._base_model import BaseModel
from src.models.loss import AlphaFoldLoss
from src.models.trunk import FoldingTrunk, FoldingTrunkConfig


@dataclass
class ESMFoldConfig:
    trunk: Any = field(default_factory=FoldingTrunkConfig)
    lddt_head_hid_dim: int = 128


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["s_z"])
        aux_out["distogram_logits"] = distogram_logits

        return aux_out


class ModifiedESMFoldInterpolation(BaseModel):
    def __init__(
        self,
        use_embedding=True,
        use_edge_transition=True,
        esmfold_config=None,
        deterministic=False,
        is_forecast=False,
        **kwargs,
    ):
        """
        ref: https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py

        Args:
            use_embedding (bool): whether to use an embedding layer to generate initial single/pair representation passed
                to the IPA module. When false, use the default FolddingTrunk as in ESMfold. Note that recycling is diabled
                when using the embedding layer.
            use_edge_transition (bool): whether to update pair representation in IPA moudle.
            esmfold_config: custom ESMfold configuration.
            deterministic (bool): if False, use an uncertainty prediction module to fit the covariance.
            is_forecast (bool): whether to initialize as a forecast network.
        """
        kwargs["deterministic"] = deterministic
        super().__init__(**kwargs)

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig()
        self.horizon = getattr(self.datamodule_config, "horizon")
        self.deterministic = deterministic
        self.is_forecast = is_forecast
        cfg = self.cfg

        self.distogram_bins = 64

        # self.esm, self.esm_dict = esm_registry.get(cfg.esm_type)()
        self.esm_dict = esm.data.Alphabet.from_architecture("ESM-1b")  # alphabet for ESM-2

        # self.esm.requires_grad_(False)
        # self.esm.half()

        self.esm_feats_protein = 1280

        # self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ModifiedESMFoldInterpolation._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(36 + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm_feats_protein),
            nn.Linear(self.esm_feats_protein, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=self.pad_idx)

        self.trunk = FoldingTrunk(
            horizon=self.horizon,
            use_embedding=use_embedding,
            use_edge_transition=use_edge_transition,
            deterministic=self.deterministic,
            is_forecast=self.is_forecast,
            **asdict(cfg.trunk),
        )

        self.openfold_cfg = config
        self.aux_heads = AuxiliaryHeads(getattr(self.openfold_cfg, "model.heads"))
        # self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        # self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        # self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        # self.lddt_bins = 50
        # self.lddt_head = nn.Sequential(
        #     nn.LayerNorm(cfg.trunk.structure_module.c_s),
        #     nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
        #     nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
        #     nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        # )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]  # type: ignore

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def forward(
        self,
        inputs: Dict,
        time: torch.Tensor,
        residx: Optional[torch.Tensor] = None,
        num_recycles: Optional[int] = None,
        **kwargs,
    ):
        """Runs a forward pass given input tokens.

        Args:
            input (Dict): Dict containing multiple input features.
            time (torch.Tensor): interpolation timestep.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles.
        """

        r1 = dict()
        r2 = dict()
        for key, value in inputs.items():
            r1[key] = value[0]
            r2[key] = value[1]

        mask = r2["backbone_rigid_mask"]

        rigids1 = Rigid.from_tensor_4x4(r1["backbone_rigid_tensor"])
        rigids2 = Rigid.from_tensor_4x4(r2["backbone_rigid_tensor"])

        aa_shifted = r1["protein_aatype"]  # torch.Size([batch_size, L_{aa}]) shifted 1 to leave 0 for padding & mask
        aa = r1["aatype"]  # true aa
        esm_s_protein = r1["esm"]  # torch.Size([batch_size, L_{aa}, 1280])

        types = aa_shifted  # [B, L_{aa}]
        true_types = aa

        B = types.shape[0]
        L = types.shape[1]
        device = types.device

        if residx is None:  # generate residx containing ligand atoms
            residx = torch.arange(L, device=device).expand_as(types)

        dssp = r1["dssp"]

        # === preprocessing ===
        # weigh representations from different layers
        # esm_s_protein = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s_protein).squeeze(2)

        s_s_0_protein = self.esm_s_mlp(esm_s_protein)  # torch.Size([batch_size, num_res, c_s])

        s_s_0 = s_s_0_protein  # torch.Size([batch_size, seq_length, c_s])

        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
        s_s_0 += self.embedding(types)

        structure: dict = self.trunk(
            s_s_0,
            s_z_0,
            rigids1,
            rigids2,
            dssp,
            time,
            true_types,
            residx,
            mask,
            no_recycles=num_recycles,
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
                "single",
                "bb_update_weight",
                # ---------- uncertainty related outputs
                "covariance_p",
                "scale_tril_p",
                "frames_uncrty",
                "sidechain_frames_uncrty",
                "unnormalized_angles_uncrty",
                "angles_uncrty",
                "positions_uncrty",
                "bb_update_weight_uncrty",
            ]
        }

        # lm_logits = self.lm_head(structure["s_s"])
        # structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= r2["backbone_rigid_mask"].unsqueeze(-1)

        structure.update(self.aux_heads(structure))

        structure["final_atom_positions"] = atom14_to_atom37(structure["positions"][-1], structure)
        structure["final_atom_mask"] = structure["atom37_atom_exists"]
        structure["final_affine_tensor"] = structure["frames"][-1]

        # ------ uncertainty related outputs
        if not self.deterministic:
            structure["final_atom_positions_uncrty"] = atom14_to_atom37(structure["positions_uncrty"][-1], structure)
            structure["final_affine_tensor_uncrty"] = structure["frames_uncrty"][-1]

        return structure

    def get_loss(
        self,
        inputs: dict,
        targets: dict,
        time: Tensor,
        metadata: Any = None,
        return_predictions: bool = False,
        **kwargs,
    ):
        structure = self(inputs, time)

        if isinstance(self.criterion, AlphaFoldLoss):
            cumulative_loss, losses = self.criterion(structure, targets)
            loss = {f"train/{k}": v for k, v in losses.items()}
            loss.update({"loss": cumulative_loss})
        else:
            raise RuntimeError(f"Unknown criterion {self.criterion} for ModifiedESMFold.")

        return loss


class ModifiedESMFoldForecast(ModifiedESMFoldInterpolation):
    def __init__(
        self,
        use_embedding=True,
        use_edge_transition=True,
        esmfold_config=None,
        deterministic=True,
        with_time_emb=True,
        **kwargs,
    ):
        """
        Args:
            use_embedding (bool): whether to use an embedding layer to generate initial single/pair representation passed
                to the IPA module. When false, use the default FolddingTrunk as in ESMfold. Note that recycling is diabled
                when using the embedding layer.
            use_edge_transition (bool): whether to update pair representation in IPA moudle.
            esmfold_config: custom ESMfold configuration.
            deterministic: if False, use an uncertainty prediction module to fit the covariance.
        """
        assert deterministic is True, "Forecast network only supports `deterministic=True`."
        super().__init__(use_embedding, use_edge_transition, esmfold_config, deterministic, is_forecast=True, **kwargs)

    def forward(self, inputs: Dict, time: Tensor, condition: Sequence[Dict]):
        """
        Args:
            inputs (dict): x_t, the interpolated data
            time (tensor): time infomation
            condition ((dict, dict)): contains x_0 and the static condition,
                i.e., the initial conditions at time t0 and the aatype, mask, esm infomation.
        """
        initial_condition, static_condition = condition
        inputs = {**inputs, **static_condition}
        initial_condition = {**initial_condition, **static_condition}

        inputs = {k: v for k, v in inputs.items() if k in initial_condition.keys()}

        forecast_input = dict()
        for key in inputs.keys():
            assert key in initial_condition.keys(), "inputs and initial_condition does not match."
            forecast_input[key] = [inputs[key], initial_condition[key]]
        structure = super().forward(forecast_input, time)
        return structure
