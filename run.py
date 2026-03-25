import os

import hydra
from omegaconf import DictConfig

from src.train import run_inference


@hydra.main(config_path="src/configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """Run inference based on the config file configs/main_config.yaml (and any command-line overrides)."""
    return run_inference(config)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
