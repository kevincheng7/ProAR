from __future__ import annotations

import os
import re
from typing import Dict, Optional

import mdtraj as md
import numpy as np
import torch
from torch.utils import data

from openfold.config import config
from openfold.data import data_transforms
from openfold.data.data_pipeline import make_pdb_features
from openfold.np import protein
from src.utilities.utils import get_logger


# For more information, see https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html
DSSP_MAP = {
    "H": 1,  # Helix. Either of the 'H', 'G', or 'I' codes.
    "E": 2,  # Strand. Either of the 'E', or 'B' codes.
    "C": 3,  # Coil. Either of the 'T', 'S' or ' ' codes.
}

log = get_logger(__name__)


def nonensembled_transform_fns():
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.squeeze_features,
        data_transforms.make_seq_mask,
    ]
    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )
    transforms.extend(
        [
            data_transforms.make_atom14_positions,
            data_transforms.atom37_to_frames,
            data_transforms.atom37_to_torsion_angles(""),
            data_transforms.make_pseudo_beta(""),
            data_transforms.get_backbone_frames,
            data_transforms.get_chi_angles,
        ]
    )
    crop_feats = dict(getattr(config, "data.common.feat"))
    transforms.append(data_transforms.select_feat(list(crop_feats)))  # type: ignore
    return transforms


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def process_pdb_string(pdb_str):
    processed_lines = []
    residue_mapping = {
        "HIE": "HIS",
        "HID": "HIS",
        "HIP": "HIS",
        "HSE": "HIS",
        "HSD": "HIS",
        "HSP": "HIS",
        "ASH": "ASP",
        "GLH": "GLU",
        "LYN": "LYS",
        "CYX": "CYS",
    }

    for line in pdb_str.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            residue_name = line[17:20].strip()
            if residue_name in residue_mapping:
                line = line[:17] + residue_mapping[residue_name] + line[20:]

            # Delete insertion code which is used when two residues have the same numbering.
            line = line[:26] + " " + line[27:]

        processed_lines.append(line)

    return "\n".join(processed_lines)


def np_example_to_features(np_example: Dict):
    feature_names = getattr(config, "data.common.unsupervised_features")
    feature_names += getattr(config, "data.supervised.supervised_features")
    feature_names += [
        "protein_aatype",
    ]

    tensor_dict = {k: torch.tensor(v) for k, v in np_example.items() if k in feature_names}

    with torch.no_grad():
        nonensembled = nonensembled_transform_fns()
        tensors = compose(nonensembled)(tensor_dict)  # type: ignore

    return tensors


def process_protein_pdb(
    pdb_path: Optional[str] = None,
    mean_coords: Optional[np.ndarray] = None,
    pdb_str: Optional[str] = None,
    protein_object: Optional[protein.Protein] = None,
    is_distillation: bool = False,
    chain_id: Optional[str] = None,
) -> Dict:
    """Assembles features for a protein in a PDB file."""
    if protein_object is None:
        if pdb_str is None:
            if pdb_path is None:
                raise ValueError("At least one of `protein_object`, `pdb_str`, or `pdb_path` must be provided.")

            with open(pdb_path, "r") as f:
                pdb_str = f.read()

        pdb_str = process_pdb_string(pdb_str)
        pdb_str = re.sub(r"(^.{21})([^A-Z])", r"\1A", pdb_str, flags=re.MULTILINE)

        protein_object = protein.from_pdb_string(pdb_str, chain_id)

    if mean_coords is None:
        mean_coords = np.zeros(3)

    atom_positions = protein_object.atom_mask[..., None] * (protein_object.atom_positions - mean_coords)
    protein_object = protein.Protein(
        atom_positions=atom_positions,
        aatype=protein_object.aatype,
        atom_mask=protein_object.atom_mask,
        residue_index=protein_object.residue_index,
        b_factors=protein_object.b_factors,
        chain_index=protein_object.chain_index,
        remark=protein_object.remark,
        parents=protein_object.parents,
        parents_chain_index=protein_object.parents_chain_index,
    )

    pdb_feats = make_pdb_features(protein_object, "", is_distillation=is_distillation)
    pdb_feats = {
        k: v
        for k, v in pdb_feats.items()
        if k
        in [
            "aatype",              # (num_res, num_unique_aas) one-hot
            "residue_index",       # (num_res,)
            "seq_length",          # (num_res,) of same value
            "sequence",            # (1,) protein sequence string
            "all_atom_positions",  # (num_res, num_atom_type, 3)
            "all_atom_mask",       # (num_res, num_atom_type)
        ]
    }
    pdb_feats["protein_aatype"] = (
        protein_object.aatype + 1
    )  # (num_res,) leave 0 for padding, range is [0, 21] where 21 is 'X'

    pdb_feats = np_example_to_features(pdb_feats)

    return pdb_feats


def compute_center_of_mass(protein_path: str):
    """Compute the center of mass of a protein structure."""
    with open(protein_path, "r") as f:
        pdb_str = f.read()
        pdb_str = process_pdb_string(pdb_str)
        pdb_str = re.sub(r"(^.{21})([^A-Z])", r"\1A", pdb_str, flags=re.MULTILINE)

    protein_object = protein.from_pdb_string(pdb_str, chain_id=None)

    valid_protein_coords = protein_object.atom_positions[protein_object.atom_mask == 1]
    mean_coords = np.mean(valid_protein_coords, axis=0)

    return mean_coords, protein_object


class InferenceDataset(data.Dataset):
    """
    Simple inference dataset that loads proteins from a directory structure::

        data_dir/
          protein_A/
            init.pdb            # Initial frame PDB file
            esm_seq.npy         # (num_res, 1280) ESM sequence representation
            esm_pair.npy        # (num_res, num_res, 20) ESM pair representation
          protein_B/
            ...

    Each subdirectory is one protein. The directory name is used as the protein identifier.

    Args:
        data_dir: Root directory containing protein subdirectories.
        max_residues: If set, skip proteins with more residues than this limit.
    """

    def __init__(self, data_dir: str, max_residues: Optional[int] = None):
        super().__init__()
        self.data_dir = data_dir
        self.max_residues = max_residues

        candidates = sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )

        self.proteins = []
        self.sizes = {}
        for name in candidates:
            pdb_path = os.path.join(data_dir, name, "init.pdb")
            esm_seq_path = os.path.join(data_dir, name, "esm_seq.npy")
            esm_pair_path = os.path.join(data_dir, name, "esm_pair.npy")

            if not os.path.isfile(pdb_path):
                log.warning(f"Skipping {name}: missing init.pdb")
                continue
            if not os.path.isfile(esm_seq_path):
                log.warning(f"Skipping {name}: missing esm_seq.npy")
                continue
            if not os.path.isfile(esm_pair_path):
                log.warning(f"Skipping {name}: missing esm_pair.npy")
                continue

            if max_residues is not None:
                num_res = np.load(esm_seq_path).shape[0]
                if num_res > max_residues:
                    log.info(f"Skipping {name}: {num_res} residues > max_residues={max_residues}")
                    continue
                self.sizes[name] = num_res

            self.proteins.append(name)

        log.info(f"InferenceDataset: found {len(self.proteins)} proteins in {data_dir}")

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        name = self.proteins[index]
        protein_dir = os.path.join(self.data_dir, name)

        pdb_path = os.path.join(protein_dir, "init.pdb")
        esm_seq_path = os.path.join(protein_dir, "esm_seq.npy")
        esm_pair_path = os.path.join(protein_dir, "esm_pair.npy")

        mean_coords, protein_object = compute_center_of_mass(pdb_path)
        pdb_feats = process_protein_pdb(protein_object=protein_object, mean_coords=mean_coords)

        esm = torch.tensor(np.load(esm_seq_path))       # (num_res, 1280)
        esm_pair = torch.tensor(np.load(esm_pair_path))  # (num_res, num_res, 20)

        frame = md.load(pdb_path)
        dssp_raw = md.compute_dssp(frame)[0]
        dssp = torch.tensor([DSSP_MAP[ss] for ss in dssp_raw], dtype=torch.long)  # (num_res,)

        feats = {**pdb_feats, "esm": esm, "esm_pair": esm_pair, "dssp": dssp}

        # Add window dimension (window=1) so shape is (1, *) per feature
        result = {k: v.unsqueeze(0) for k, v in feats.items()}
        result["meta_data"] = [{"system": name, "time": 0}]

        return result
