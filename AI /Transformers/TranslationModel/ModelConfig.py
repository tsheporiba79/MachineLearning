from typing import Dict, Union
from pathlib import Path


def get_config() -> Dict[str, Union[int, str]]:
    """
    Get configuration parameters.

    Returns:
        dict: Configuration parameters.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_source": "en",
        "lang_target": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config: Dict[str, Union[int, str]], epoch: str) -> str:
    """
    Get the path to the weights file for a specific epoch.

    Args:
        config (dict): Configuration parameters.
        epoch (str): Epoch number.

    Returns:
        str: Path to the weights file.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config: Dict[str, Union[int, str]]) -> Union[str, None]:
    """
    Get the path to the latest weights file.

    Args:
        config (dict): Configuration parameters.

    Returns:
        str or None: Path to the latest weights file if found, otherwise None.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
