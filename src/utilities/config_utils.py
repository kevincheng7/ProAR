import os
import warnings
from typing import List, Sequence, Union

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

from src.utilities.utils import get_logger


log = get_logger(__name__)


@rank_zero_only
def print_config(
    config,
    fields: Union[str, Sequence[str]] = (
        "datamodule",
        "model",
        "trainer",
        "seed",
    ),
    resolve: bool = True,
    rich_style: str = "magenta",
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure (if installed: ``pip install rich``).

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        rich_style (str, optional): Style of Rich library to use for printing. E.g "magenta", "bold", "italic", etc.
    """
    import importlib

    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        print(OmegaConf.to_yaml(config, resolve=resolve))
        return
    import rich.syntax
    import rich.tree

    tree = rich.tree.Tree(":gear: CONFIG", style=rich_style, guide_style=rich_style)
    if isinstance(fields, str):
        if fields.lower() == "all":
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=rich_style, guide_style=rich_style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def extras(config: DictConfig, allow_permission_error: bool = False) -> DictConfig:
    """Optional utilities controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    - checking if config values are valid

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        allow_permission_error (bool): Whether to allow PermissionError when creating working dir.
    """
    if config.get("work_dir"):
        try:
            os.makedirs(name=config.get("work_dir"), exist_ok=True)
        except PermissionError as e:
            if allow_permission_error:
                log.warning(f"PermissionError: {e}")
            else:
                raise e

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug_mode"):
        log.info("Running in debug mode! <config.debug_mode=True>")

        OmegaConf.set_struct(config, False)
        config.trainer.fast_dev_run = 100
        if "logger" in config:
            del config.logger
        if "callbacks" in config:
            del config.callbacks
        if config.get("module") and config.module.get("monitor"):
            del config.module.monitor
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["OC_CAUSE"] = "1"
        torch.autograd.set_detect_anomaly(True)

    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("devices"):
            config.trainer.devices = 1
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
    elif config.datamodule.get("num_workers") == -1:
        config.datamodule.num_workers = os.cpu_count()
        log.info(f"Automatically setting num_workers to {config.datamodule.num_workers} (CPU cores).")

    strategy = config.trainer.get("strategy", "")
    if strategy.startswith("ddp") or strategy.startswith("dp"):
        if config.datamodule.get("pin_memory"):
            log.info(f"Forcing ddp friendly configuration! <config.trainer.strategy={strategy}>")
            config.datamodule.pin_memory = False

    torch_matmul_precision = config.get("torch_matmul_precision", "highest")
    if torch_matmul_precision != "highest":
        log.info(f"Setting torch matmul precision to ``{torch_matmul_precision}``.")
        torch.set_float32_matmul_precision(torch_matmul_precision)

    bs = config.datamodule.get("batch_size", 1)
    acc = config.trainer.get("accumulate_grad_batches", 1)
    n_gpus = config.trainer.get("devices", 1)
    if n_gpus == "auto":
        n_gpus = torch.cuda.device_count()
    elif isinstance(n_gpus, str) and "," in n_gpus:
        n_gpus = len(n_gpus.split(","))
    elif isinstance(n_gpus, Sequence):
        n_gpus = len(n_gpus)

    with open_dict(config):
        config.n_gpus = n_gpus
        config.effective_batch_size = bs * acc * n_gpus

    if not torch.cuda.is_available():
        if config.trainer.get("accelerator") == "gpu":
            config.trainer.accelerator = "cpu"
            log.warning(
                "CUDA is not available, switching to CPU.\n"
                "\tIf you want to use GPU, please re-install pytorch: https://pytorch.org/get-started/locally/."
                "\n\tIf you want to use a different accelerator, specify it with ``trainer.accelerator=...``."
            )

    try:
        _ = config.datamodule.get("data_dir")
    except omegaconf.errors.InterpolationResolutionError as e:
        raise ValueError(
            "Could not resolve ``datamodule.data_dir`` in config. See error message above for details.\n"
            "   If this is a Windows machine, you may need to set ``data_dir`` to an absolute path, e.g. ``C:/data``.\n"
            "       You can do so in ``src/configs/datamodule/_base_data_config.yaml`` or with the command line."
        ) from e

    if config.module.get("num_predictions", 1) > 1:
        is_ipol_exp = "InterpolationExperiment" in config.module.get("_target_", "")
        monitor = config.module.get("monitor", "") or ""
        if "crps" not in monitor:
            new_monitor = f"val/{config.module.num_predictions}ens_mems/"
            new_monitor += "ipol/avg/crps" if is_ipol_exp else "avg/crps"
            config.module.monitor = new_monitor
            log.info(f"Setting monitor to {new_monitor} since num_predictions > 1")

    monitor = config.module.get("monitor", "") or ""
    if monitor:
        clbk_ckpt = config.callbacks.get("model_checkpoint", None)
        clbk_es = config.callbacks.get("early_stopping", None)
        if clbk_ckpt is not None and clbk_ckpt.get("monitor"):
            config.callbacks.model_checkpoint.monitor = monitor
        if clbk_es is not None and clbk_es.get("monitor"):
            config.callbacks.early_stopping.monitor = monitor

    model_name = config.model.get("name")
    if model_name is None or model_name == "":
        model_class = config.model.get("_target_", "")
        config.model.name = model_class.rsplit(".", 1)[-1] if model_class else "unknown"

    check_config_values(config)

    if config.get("print_config"):
        print_fields = ("model", "diffusion", "datamodule", "module", "trainer", "seed", "work_dir")
        print_config(config, fields=print_fields)
        if config.get("callbacks") and config.callbacks.get("early_stopping"):
            patience = config.callbacks.early_stopping.get("patience")
            monitor = config.callbacks.early_stopping.get("monitor")
            mode = config.callbacks.early_stopping.get("mode")
            log.info(f"Early stopping: patience={patience}, monitor={monitor}, mode={mode}")

    return config


def check_config_values(config: DictConfig):
    """Check if config values are valid."""
    with open_dict(config):
        if "net_normalization" in config.model.keys():
            if config.model.net_normalization is None:
                config.model.net_normalization = "none"
            config.model.net_normalization = config.model.net_normalization.lower()

        if config.get("diffusion", default_value=False):
            for k, v in config.model.items():
                if k in config.diffusion.keys() and k not in ["_target_", "name"]:
                    assert v == config.diffusion[k], f"Diffusion model and model have different values for {k}!"

        scheduler_cfg = config.module.get("scheduler")
        if scheduler_cfg and "LambdaWarmUpCosineScheduler" in scheduler_cfg._target_:
            config.module.optimizer.lr = 1.0

        if config.module.get("num_predictions", 1) > 1:
            ebs = config.datamodule.get("eval_batch_size", 1)
            effective_ebs = ebs * config.module.num_predictions
            log.info(
                f"Note that the effective evaluation batch size will be multiplied by the number of "
                f"predictions={config.module.num_predictions} for a total of {effective_ebs}!"
            )


def get_all_instantiable_hydra_modules(config, module_name: str):
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                if "early_stopping" in module_config.get("_target_"):
                    diffusion = config.get("diffusion", default_value=False)
                    monitor = module_config.get("monitor", "")
                    if diffusion and "step" not in monitor and "epoch" not in monitor:
                        module_config.monitor += "_step"
                        log.info(f"Early stopping monitor changed to: {module_config.monitor}")

                try:
                    modules.append(hydra.utils.instantiate(module_config))
                except omegaconf.errors.InterpolationResolutionError as e:
                    log.warning(f" Hydra could not instantiate {module_config} for module_name={module_name}")
                    raise e

    return modules


@rank_zero_only
def log_hyperparameters(*args, **kwargs) -> None:
    """No-op: hyperparameter logging is not needed for inference."""
    pass


def get_config_from_hydra_compose_overrides(
    overrides: List[str],
    config_path: str = "../configs",
    config_name: str = "main_config.yaml",
) -> DictConfig:
    """
    Function to get a Hydra config manually based on a default config file and a list of override strings.
    This is an alternative to using hydra.main(..) and the command-line for overriding the default config.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.
        config_path: Relative path to the folder where the default config file is located.
        config_name: Name of the default config file (.yaml ending).

    Returns:
        The resulting config object based on the default config file and the overrides.

    Examples:

    .. code-block:: python

        config = get_config_from_hydra_compose_overrides(overrides=['model=mlp', 'model.optimizer.lr=0.001'])
        print(f"Lr={config.model.optimizer.lr}, MLP hidden_dims={config.model.hidden_dims}")
    """
    from hydra.core.global_hydra import GlobalHydra

    overrides = list(set(overrides))
    if "-m" in overrides:
        overrides.remove("-m")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, version_base=None)
    try:
        config = hydra.compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
    return config


def get_model_from_hydra_compose_overrides(overrides: List[str]):
    """
    Function to get a torch model manually based on a default config file and a list of override strings.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.

    Returns:
        The model instantiated from the resulting config.

    Examples:

    .. code-block:: python

        mlp_model = get_model_from_hydra_compose_overrides(overrides=['model=mlp'])
        random_mlp_input = torch.randn(1, 100)
        random_prediction = mlp_model(random_mlp_input)
    """
    from src.interface import get_lightning_module

    cfg = get_config_from_hydra_compose_overrides(overrides)
    return get_lightning_module(cfg)
