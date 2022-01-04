import os
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from box import Box
from models.dataset import CustomDataModule, CustomDataset
from models.model import Model, inference
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


##########################
# Setup
##########################
def load_config(config_path: str) -> Box:
    """設定の読み込み

    Args:
        config_path (str): 設定ファイルへのパス

    Returns:
        Box: 設定
    """
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    config = Box(config)
    return config


def setup(config_path: str) -> Box:
    """初期設定

    Args:
        config_path (str): 設定ファイルへのパス

    Returns:
        Box: 設定
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(config_path)

    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)
    return config


##########################
# DataLoad
##########################
def load_raw_data(config: Box) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生データをロード

    Args:
        config (Box): 設定

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 訓練データとテストデータのメタ情報
    """
    input_dir = Path(config.input_dir)

    train_df = pd.read_table(input_dir / "train_annotations.tsv")
    test_df = pd.DataFrame(
        list(map(lambda x: x.stem, input_dir.glob("test/*"))), columns=["id"]
    )
    train_df["path"] = train_df["id"].apply(
        lambda x: str(input_dir / "train" / x) + ".png"
    )
    test_df["path"] = test_df["id"].apply(
        lambda x: str(input_dir / "test" / x) + ".png"
    )
    return train_df, test_df


def load_pseudo_df(config: Box) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """疑似ラベルデータをロード

    Args:
        config (Box): 設定

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 疑似ラベルの貼られた訓練データ，テストデータのメタ情報
    """
    input_dir = Path(config.input_dir)
    pseudo_dir = Path(config.pseudo_dir)

    train_pseudo_df = pd.read_csv(pseudo_dir / "train_pseudo.csv")
    test_pseudo_df = pd.read_csv(pseudo_dir / "test_pseudo.csv")
    train_pseudo_df["path"] = train_pseudo_df["id"].apply(
        lambda x: str(input_dir / "train" / x) + ".png"
    )
    test_pseudo_df["path"] = test_pseudo_df["id"].apply(
        lambda x: str(input_dir / "test" / x) + ".png"
    )
    return train_pseudo_df, test_pseudo_df


def load_all_data(config: Box) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """学習用のメタデータをロード

    Args:
        config (Box): config

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 学習データと疑似ラベルのメタ情報
    """
    train_df, _ = load_raw_data(config)
    train_pseudo_df, test_pseudo_df = load_pseudo_df(config)

    train_df = train_df.merge(train_pseudo_df[["id", "texture"]], how="left", on="id")
    return train_df, test_pseudo_df


##########################
# Training
##########################
def cross_validation(
    base_df: pd.DataFrame, pseudo_df: pd.DataFrame, threshold: int, config: Box
) -> pd.DataFrame:
    """交差検証

    threshold percentile以上を異常値とする疑似ラベルを作成

    Args:
        base_df (pd.DataFrame): 訓練データのメタ情報
        pseudo_df (pd.DataFrame): 疑似ラベルのメタ情報
        threshold (int): 何パーセンタイル点以上を異常値とするか
        config (Box): 設定

    Returns:
        pd.DataFrame: 疑似ラベルのメタ情報
    """
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    df = pd.concat([base_df, pseudo_df], ignore_index=True)
    pseudo_preds = np.zeros(len(pseudo_df))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df["id"], df["label"].astype("str") + df["texture"].astype(str))
    ):
        print("**" * 10, f"THRESHOLD={threshold}, FOLD={fold}", "**" * 10)
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = CustomDataModule(train_df, val_df, config)
        pseudo_dataset = CustomDataset(
            pseudo_df[["path"]], image_size=config.transform.image_size
        )
        pseudo_dataloader = DataLoader(pseudo_dataset, **config.val_loader)

        model = Model(config)
        earystopping = EarlyStopping(**config.earlystopping)
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(**config.ckpt)
        logger = TensorBoardLogger(
            f"../output/{config.exp_name}/{config.model.name}/thresh{threshold}",
            version=fold,
        )

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)
        pseudo_preds += inference(model, pseudo_dataloader)

    # pseudo-label each texture.
    pseudo_preds /= config.n_splits
    pseudo_df["label"] = None
    for texture in pseudo_df["texture"].unique():
        texture_idx = pseudo_df.query("texture == @texture").index
        pseudo_preds_i = pseudo_preds[texture_idx]
        percentile = np.percentile(pseudo_preds_i, threshold)
        pseudo_df.loc[texture_idx, "label"] = (pseudo_preds_i > percentile).astype(int)
    return pseudo_df


@click.command()
@click.option("--config_path", "-c", default="../config/train_cfg.yaml")
def main(config_path):
    # setup
    config = setup(config_path)

    # dataload
    train_df, pseudo_df = load_all_data(config)
    train_df = train_df.head(20)
    pseudo_df = pseudo_df.head(20)

    # training 閾値を小さくしながら疑似ラベルを蒸留して学習
    for threshold in config.thresholds:
        pseudo_df = cross_validation(train_df, pseudo_df, threshold, config)


if __name__ == "__main__":
    main()
