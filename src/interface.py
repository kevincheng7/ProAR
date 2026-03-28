from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.datamodules.abstract_datamodule import BaseDataModule
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import get_logger, rename_state_dict_keys_and_save


log = get_logger(__name__)


def get_lightning_module(config: DictConfig, **kwargs) -> BaseExperiment:
    """Get the lightning module as defined by the key value pairs in ``config.module``."""
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        diffusion_config=config.get("diffusion", default_value=None),
        _recursive_=False,
        **kwargs,
    )
    return model


def get_datamodule(config: DictConfig) -> BaseDataModule:
    """Get the datamodule as defined by the key value pairs in ``config.datamodule``."""
    data_module = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig):
    """Get both the model and datamodule from config."""
    data_module = get_datamodule(config)
    model = get_lightning_module(config)
    return model, data_module


def reload_model_from_config_and_ckpt(
    config: DictConfig,
    model_path: str,
    device: Optional[torch.device] = None,
    also_datamodule: bool = True,
    also_ckpt: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Load a model as defined by ``config.model`` and reload its weights from ``model_path``."""
    model, data_module = get_model_and_data(config) if also_datamodule else (get_lightning_module(config), None)
    model_state = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = rename_state_dict_keys_and_save(model_state, model_path)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.interpolator")}
    model.load_state_dict(state_dict, strict=False)

    to_return = {
        "model": model,
        "datamodule": data_module,
        "epoch": model_state.get("epoch", 0),
        "global_step": model_state.get("global_step", 0),
    }
    if also_ckpt:
        to_return["ckpt"] = model_state
    return to_return


def get_checkpoint_from_path(
    model_checkpoint: Optional[torch.nn.Module] = None,
    model_checkpoint_path: Optional[str] = None,
    hydra_config_path: Optional[str] = None,
) -> torch.nn.Module:
    """Load a model checkpoint from a local path or return the provided module."""
    if model_checkpoint is not None:
        return model_checkpoint
    elif model_checkpoint_path is not None:
        assert hydra_config_path is not None, "must provide hydra_config_path when specifying model_checkpoint_path"
        override_key_value = ["module.verbose=False"]
        overrides = [o for o in override_key_value if "=" in o and "." in o.split("=")[0]]

        config = OmegaConf.load(hydra_config_path)
        overrides = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.unsafe_merge(config, overrides)

        try:
            model = reload_model_from_config_and_ckpt(config, model_checkpoint_path, also_datamodule=False)["model"]
            log.info(f"Successfully loaded model checkpoint from local path {model_checkpoint_path}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load model checkpoint from {model_checkpoint_path}. "
                f"The model code may be incompatible with the checkpoint.\n{e}"
            )
        return model
    else:
        raise ValueError("Provide either model_checkpoint or model_checkpoint_path (with hydra_config_path)")
