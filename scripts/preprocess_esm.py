#!/usr/bin/env python
"""
Generate ESM-2 sequence and pair representations for ProAR inference.

For each protein subdirectory containing ``init.pdb``, this script:
  1. Extracts the amino-acid sequence from the PDB file.
  2. Runs ESM-2 (esm2_t33_650M_UR50D) to compute:
       - esm_seq.npy  : per-residue representation, shape (num_res, 1280)
       - esm_pair.npy : last-layer attention map,     shape (num_res, num_res, 20)
  3. Saves both arrays into the same subdirectory.

Usage:
    python scripts/preprocess_esm.py --data_dir ./data/atlas_test [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys

import esm
import numpy as np
import torch
from Bio.PDB import PDBParser


REPR_LAYER = 33
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Non-standard mappings that may appear in MD PDBs
    "HIE": "H", "HID": "H", "HIP": "H", "HSE": "H", "HSD": "H", "HSP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "CYX": "C",
}


def extract_sequence(pdb_path: str) -> str:
    """Extract the amino-acid sequence from a PDB file (first model, first chain)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())

    residues = []
    for res in chain:
        resname = res.get_resname().strip()
        if resname in THREE_TO_ONE:
            residues.append((res.id[1], THREE_TO_ONE[resname]))

    residues.sort(key=lambda x: x[0])
    return "".join(r[1] for r in residues)


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 features for ProAR")
    parser.add_argument("--data_dir", required=True, help="Root directory with protein subdirectories")
    parser.add_argument("--device", default=None, help="Device (cuda / cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Discover proteins
    proteins = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.isfile(os.path.join(data_dir, d, "init.pdb"))
    )
    print(f"Found {len(proteins)} proteins in {data_dir}")

    # Filter to those needing processing
    to_process = []
    for name in proteins:
        seq_path = os.path.join(data_dir, name, "esm_seq.npy")
        pair_path = os.path.join(data_dir, name, "esm_pair.npy")
        if os.path.isfile(seq_path) and os.path.isfile(pair_path):
            continue
        to_process.append(name)

    if not to_process:
        print("All proteins already have ESM features. Nothing to do.")
        return

    print(f"{len(to_process)} proteins need ESM features. Loading model ...")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    for idx, name in enumerate(to_process):
        pdb_path = os.path.join(data_dir, name, "init.pdb")
        seq_out = os.path.join(data_dir, name, "esm_seq.npy")
        pair_out = os.path.join(data_dir, name, "esm_pair.npy")

        try:
            sequence = extract_sequence(pdb_path)
        except Exception as e:
            print(f"[{idx+1}/{len(to_process)}] {name} -- ERROR extracting sequence: {e}, skipping")
            continue

        if not sequence:
            print(f"[{idx+1}/{len(to_process)}] {name} -- WARNING: empty sequence, skipping")
            continue

        num_res = len(sequence)
        print(f"[{idx+1}/{len(to_process)}] {name}  (L={num_res}) ...", end=" ", flush=True)

        _, _, batch_tokens = batch_converter([(name, sequence)])
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[REPR_LAYER], return_contacts=True)

        token_repr = results["representations"][REPR_LAYER]  # (1, seq+2, 1280)
        attentions = results["attentions"]  # (1, layers, heads, seq+2, seq+2)

        tokens_len = batch_lens[0].item()
        esm_seq = token_repr[0, 1:tokens_len - 1].cpu().numpy()  # (num_res, 1280)
        esm_pair = (
            attentions[0, REPR_LAYER - 1, :, 1:tokens_len - 1, 1:tokens_len - 1]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )  # (num_res, num_res, 20)

        assert esm_seq.shape == (num_res, 1280), f"esm_seq shape mismatch: {esm_seq.shape}"
        assert esm_pair.shape == (num_res, num_res, 20), f"esm_pair shape mismatch: {esm_pair.shape}"

        np.save(seq_out, esm_seq)
        np.save(pair_out, esm_pair)
        print("done")

    print(f"\nAll done. Features saved under {data_dir}")


if __name__ == "__main__":
    main()
