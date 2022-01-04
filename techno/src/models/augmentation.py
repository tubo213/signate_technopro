import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485]
IMAGENET_STD = [0.229]


class MyRotateTransform:
    """指定した角度からランダムに回転"""

    def __init__(self, angles: Sequence[int], p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        if self.p <= random.random():
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)
        else:
            return x


def get_default_transforms() -> Dict[T.Compose, T.Compose]:
    """augmentationを取得

    Returns:
        Dict[T.Compose, T.Compose]: 学習用，検証用のaugmentation
    """
    transform = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                MyRotateTransform([90, 180, 270]),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }
    return transform


def mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Mixup
    ref: mixup: Beyond Empirical Risk Mnimization

    Args:
        x (torch.Tensor): 特徴量
        y (torch.Tensor): ラベル
        alpha (float, optional): Be(alpha, alpha)のalpha. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam
