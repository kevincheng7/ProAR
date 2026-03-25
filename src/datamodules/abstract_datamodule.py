import multiprocessing
from typing import Any, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader

from src.utilities.utils import get_logger, raise_error_if_invalid_value


log = get_logger(__name__)


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule for ProAR inference.

    Subclasses should implement ``setup()`` to populate ``self._data_test``
    and ``self._data_predict``, and override ``test_dataloader()`` /
    ``predict_dataloader()`` if custom collation is needed.
    """

    _data_test: Any
    _data_predict: Any

    def __init__(
        self,
        data_dir: str,
        model_config: DictConfig = None,
        eval_batch_size: int = 1,
        num_workers: int = -1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Args:
            data_dir: Path to the data folder.
            eval_batch_size: Batch size for test and prediction dataloaders.
            num_workers: Number of workers for data loading (-1 = all available CPUs).
            pin_memory: Dataloader arg for higher efficiency.
            drop_last: Drop the last incomplete batch.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_config", "verbose"])
        self.model_config = model_config
        self.test_batch_size = eval_batch_size
        self._data_test = self._data_predict = None
        self._check_args()

    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    def print_data_sizes(self, stage: str = None):
        if stage in ["test", None] and self._data_test is not None:
            log.info(f" Dataset test size: {len(self._data_test)}")
        elif stage == "predict" and self._data_predict is not None:
            log.info(f" Dataset predict size: {len(self._data_predict)}")

    def setup(self, stage: Optional[str] = None):
        raise_error_if_invalid_value(stage, ["test", "predict", None], "stage")
        raise NotImplementedError("Subclasses must implement setup().")

    @property
    def num_workers(self) -> int:
        if self.hparams.num_workers == -1:
            return multiprocessing.cpu_count()
        return int(self.hparams.num_workers)

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self._data_predict,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def boundary_conditions(
        self,
        preds: Tensor,
        targets: Tensor,
        metadata,
        time: float = None,
    ) -> Tensor:
        """Return predictions that satisfy the boundary conditions for a given item (batch element)."""
        return preds

    def get_boundary_condition_kwargs(self, batch: Any, batch_idx: int, split: str) -> dict:
        return dict(t0=0.0, dt=1.0)
