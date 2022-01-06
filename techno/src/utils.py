import os
import random

import numpy as np
import torch
import yaml
from box import Box


def seed_torch(seed: int = 1029) -> None:
    """乱数固定

    Args:
        seed (int, optional): 乱数シード. Defaults to 1029.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> Box:
    """configの読み込み

    Args:
        config_path (str): 設定ファイルへのパス

    Returns:
        Box: 設定
    """
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    config = Box(config)
    return config
