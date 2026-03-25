from omegaconf import DictConfig


def get_dims_of_dataset(datamodule_config: DictConfig):
    """Returns the number of features for the given dataset."""
    target = datamodule_config.get("_target_", datamodule_config.get("name"))
    if "molecular_dynamics_simulation" in target:
        input_dim = output_dim = spatial_dims = conditional_dim = 0
    else:
        raise ValueError(f"Unknown dataset: {target}")
    return {"input": input_dim, "output": output_dim, "spatial": spatial_dims, "conditional": conditional_dim}
