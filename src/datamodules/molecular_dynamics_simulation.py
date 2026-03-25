from __future__ import annotations

import os
from functools import partial
from itertools import chain
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, default_collate

from openfold.config import NUM_RES, config
from openfold.utils.rigid_utils import Rigid
from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.mdtrajectory import InferenceDataset
from src.utilities.utils import get_logger, raise_error_if_invalid_type


log = get_logger(__name__)

shape_schema = dict(getattr(config, "data.common.feat"))
shape_schema.update(
    {
        "esm": [NUM_RES, None],
        "esm_pair": [NUM_RES, NUM_RES, None],
        "protein_aatype": [NUM_RES],
        "dssp": [NUM_RES],
    }
)
RIGID_FEATS = ["backbone_rigid_tensor", "rigidgroups_alt_gt_frames", "rigidgroups_gt_frames"]
NOPADDING_FEATS = ["meta_data"]


def length_batching(
    batch: List[Dict[str, torch.Tensor]],
    shape_schema: Dict[str, List],
    max_squared_len: Optional[int] = None,
    use_length_batching: bool = True,
):
    """
    Customized collate_fn for variable-length proteins:
        - Pads features to the max residue count in the batch
        - Optionally controls dynamic batch size via ``max_squared_len``
    """
    if len(shape_schema["aatype"]) == 2:
        get_num_res = lambda x: x["aatype"].shape[1]
    else:
        get_num_res = lambda x: x["aatype"].shape[0]

    dicts_num_res = [(get_num_res(x), x) for x in batch]
    dicts_sorted_by_num_res = sorted(dicts_num_res, key=lambda x: x[0], reverse=True)

    max_res = dicts_sorted_by_num_res[0][0]
    pad_example = lambda x: pad_feats(x, max_res, shape_schema)

    padded_batch = [pad_example(x) for (_, x) in dicts_sorted_by_num_res]

    return default_collate(padded_batch)


def pad_feats(raw_feats: Dict, max_res: int, shape_schema: Dict[str, List]):
    """Pad sequence dimension to make fixed size."""
    pad_size_map = {NUM_RES: max_res}

    for k, v in raw_feats.items():
        if k in NOPADDING_FEATS:
            continue

        shape = list(v.shape)
        schema = shape_schema[k]
        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: {shape} vs {schema}"
        )

        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]

        if k in RIGID_FEATS:
            assert pad_size[-2:] == [4, 4], "The rigid feature should be the form of homogenous transformation."
            if pad_size != list(v.shape):
                padding = pad_size[:-2]
                padding = [((p - v.shape[i]) if (p - v.shape[i]) != 0 else p) for i, p in enumerate(padding)]
                pad_rigid = Rigid.identity(tuple(padding), dtype=v.dtype, device=v.device, requires_grad=False)  # type: ignore
                raw_feats[k] = torch.cat([v, pad_rigid.to_tensor_4x4()], dim=1)
            continue

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(chain(*padding))
        if padding:
            raw_feats[k] = torch.nn.functional.pad(v, padding)
            raw_feats[k] = torch.reshape(raw_feats[k], pad_size)

    return raw_feats


class MolecularDynamicsSimulationDataModule(BaseDataModule):
    """
    Simplified datamodule for inference on the Atlas protein dataset.

    Expects the data directory to contain protein subdirectories, each with::

        data_dir/
          protein_A/
            init.pdb            # Initial frame PDB file
            esm_seq.npy         # (num_res, 1280) ESM sequence representation
            esm_pair.npy        # (num_res, num_res, 20) ESM pair representation
          protein_B/
            ...

    Args:
        data_dir: Root directory containing protein subdirectories.
        max_squared_len: Maximum squared sequence length for GPU memory control.
    """

    def __init__(
        self,
        data_dir: str,
        max_squared_len: int = 800**2,
        **kwargs,
    ):
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        super().__init__(data_dir=data_dir, **kwargs)
        self.base_dir = os.path.expanduser(data_dir)
        self.max_squared_len = max_squared_len

        assert os.path.isdir(self.base_dir), (
            f"Could not find data directory {self.base_dir}. "
            f"Please specify the data directory using the ``datamodule.data_dir`` option."
        )
        log.info(f"Using data directory: {self.base_dir}")

    def setup(self, stage: Optional[str] = None):
        assert stage in ["test", "predict", None], f"Invalid stage '{stage}' for inference-only datamodule"
        log.info(f"Setting up MolecularDynamicsSimulationDataModule for stage {stage}...")

        max_residues = int(pow(self.max_squared_len, 0.5))
        ds = InferenceDataset(self.base_dir, max_residues=max_residues)

        if stage in ["test", None]:
            self._data_test = ds
        if stage in ["predict", None]:
            self._data_predict = ds

        if stage is not None:
            self.print_data_sizes(stage)

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(
            num_workers=self.num_workers,
            pin_memory=getattr(self.hparams, "pin_memory"),
            persistent_workers=getattr(self.hparams, "persistent_workers"),
            collate_fn=partial(
                length_batching,
                shape_schema={
                    k: [None] + v for k, v in shape_schema.items()
                },  # the first dim corresponds to the window dimension
                use_length_batching=False,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._data_predict,
            batch_size=getattr(self.hparams, "eval_batch_size"),
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )
