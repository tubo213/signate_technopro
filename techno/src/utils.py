import os
import random

import numpy as np
import torch


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
