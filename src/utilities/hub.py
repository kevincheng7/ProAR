"""HuggingFace Hub utilities for downloading ProAR model checkpoints."""

import os
from typing import Optional

from src.utilities.utils import get_logger


log = get_logger(__name__)

DEFAULT_REPO_ID = "kevincheng77/ProAR"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/proar/checkpoints")

INTERPOLATOR_FILENAME = "interpolator.ckpt"
INTERPOLATOR_CONFIG_FILENAME = "interpolator_config.yaml"
FORECASTER_FILENAME = "forecaster.ckpt"


def download_from_hub(
    filename: str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str:
    """Download a file from HuggingFace Hub.

    Args:
        filename: Name of the file to download from the repo.
        repo_id: HuggingFace Hub repository ID.
        cache_dir: Local cache directory.

    Returns:
        Local path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for automatic checkpoint downloading. "
            "Install it with: pip install huggingface_hub"
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    log.info(f"Downloaded {filename} from {repo_id} to {local_path}")
    return local_path


def get_interpolator_checkpoint(
    local_path: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Get the interpolator checkpoint path, downloading from HF Hub if needed.

    Args:
        local_path: If provided, use this local path directly.
        repo_id: HuggingFace Hub repository ID for auto-download.

    Returns:
        Path to the interpolator checkpoint file.
    """
    if local_path is not None and os.path.exists(local_path):
        return local_path
    log.info("Interpolator checkpoint not found locally, downloading from HuggingFace Hub...")
    return download_from_hub(INTERPOLATOR_FILENAME, repo_id=repo_id)


def get_interpolator_config(
    local_path: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Get the interpolator hydra config path, downloading from HF Hub if needed."""
    if local_path is not None and os.path.exists(local_path):
        return local_path
    log.info("Interpolator config not found locally, downloading from HuggingFace Hub...")
    return download_from_hub(INTERPOLATOR_CONFIG_FILENAME, repo_id=repo_id)


def get_forecaster_checkpoint(
    local_path: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Get the forecaster checkpoint path, downloading from HF Hub if needed.

    Args:
        local_path: If provided, use this local path directly.
        repo_id: HuggingFace Hub repository ID for auto-download.

    Returns:
        Path to the forecaster checkpoint file.
    """
    if local_path is not None and os.path.exists(local_path):
        return local_path
    log.info("Forecaster checkpoint not found locally, downloading from HuggingFace Hub...")
    return download_from_hub(FORECASTER_FILENAME, repo_id=repo_id)
