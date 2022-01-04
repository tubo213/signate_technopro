import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from box import Box
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import transforms as T
from tqdm import tqdm
from utils import seed_torch


##########################
# Setup
##########################
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


def setup(config_path: str) -> None:
    """初期設定

    Args:
        config_path (str): 設定ファイルへのパス
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    global CFG, INPUT_DIR, OUTPUT_DIR, DEVICE
    CFG = load_config(config_path)
    INPUT_DIR = Path(CFG.input)
    OUTPUT_DIR = Path(CFG.output)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    seed_torch(CFG.seed)


##########################
# DataLoad
##########################
def id2path(id: int, is_train: bool = True) -> str:
    """idから画像のpathを作成

    Args:
        id (int): 画像id
        is_train (bool, optional): trainかどうか. Defaults to True.

    Returns:
        str: idに対応する画像のpath
    """
    if is_train:
        return str(INPUT_DIR / "raw_data/train" / id) + ".png"
    else:
        return str(INPUT_DIR / "raw_data/test" / id) + ".png"


def get_base_df() -> pd.DataFrame:
    """メタデータのデータフレームを取得

    Returns:
        pd.DataFrame: id, path, is_trainを持ったデータフレーム
    """
    train_df = pd.read_table(INPUT_DIR / "raw_data/train_annotations.tsv")
    test_df = pd.DataFrame(
        list(map(lambda x: x.stem, INPUT_DIR.glob("raw_data/test/*"))), columns=["id"]
    )
    train_df["path"] = train_df["id"].apply(lambda id: id2path(id, is_train=True))
    test_df["path"] = test_df["id"].apply(lambda id: id2path(id, is_train=False))
    train_df["is_train"] = True
    test_df["is_train"] = False

    base_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    return base_df[["id", "path", "is_train"]]


##########################
# Preprocess
##########################
def get_img_hash(df: pd.DataFrame) -> np.ndarray:
    """CNNで画像から特徴量を取得

    Args:
        df (pd.DataFrame): メタデータ(path)の入ったデータフレーム

    Returns:
        np.ndarray: CNNで抽出した特徴量
    """
    hashes = []
    model = (
        timm.create_model("resnet34", num_classes=0, in_chans=1, pretrained=True)
        .to(DEVICE)
        .eval()
    )
    for path in tqdm(df["path"]):

        image = Image.open(path)
        image = T.ToTensor()(image).to(DEVICE).unsqueeze(0)

        image_emb = model(image)
        image_emb = image_emb.to("cpu").detach().numpy().copy()

        hashes.append(image_emb)

    return np.array(hashes).reshape(len(df), -1)


def estimate_texture(img_hash: np.ndarray) -> np.array:
    """KNNを用いてテクスチャを推定

    Args:
        img_hash (np.array): 画像の特徴量

    Returns:
        np.array: 推定したテクスチャ
    """
    km = KMeans(CFG.params.num_clusters, random_state=CFG.seed)
    km.fit(img_hash)
    return km.predict(img_hash)


def get_texture_df(df: pd.DataFrame, img_hash: np.array) -> pd.DataFrame:
    """テクスチャを含むメタデータを取得

    Args:
        df ([type]): メタデータ
        img_hash ([type]): 画像特徴量

    Returns:
        pd.DataFrame: メタデータ
    """
    texture_df = df[["id", "path", "is_train"]].copy()
    texture_df["texture"] = estimate_texture(img_hash)
    return texture_df


def estimate_pseudo_label(img_hash: np.array) -> np.array:
    """異常値推定
    GaussianMixtureで対数尤度の小さいn%を異常値とする．

    Args:
        img_hash (np.array): 画像特徴量

    Returns:
        np.array: 疑似ラベル
    """
    gm = GaussianMixture(1, random_state=CFG.seed)
    gm.fit(img_hash)

    annomaly_score = -gm.score_samples(img_hash)
    percentile = np.percentile(annomaly_score, CFG.params.threshold)
    pseudo_label = (annomaly_score >= percentile).astype(int)
    return pseudo_label


def add_pseudo_label(df: pd.DataFrame, img_hash: np.array) -> pd.DataFrame:
    """疑似ラベルを貼る

    Args:
        df (pd.DataFrame): メタデータ
        img_hash (np.array): 画像特徴量

    Returns:
        pd.DataFrame: 疑似ラベルを含むメタデータ
    """
    df["label"] = None
    for texture in df["texture"].unique():
        texture_df_i = df.query("texture == @texture")
        img_hash_i = img_hash[texture_df_i.index]
        df.loc[df["texture"] == texture, "label"] = estimate_pseudo_label(img_hash_i)

    assert df["label"].isnull().sum() == 0
    return df


##########################
# Visualize
##########################
def plot_pseudo_annomaly(df: pd.DataFrame) -> plt.Figure:
    """ランダムに16個描画

    Args:
        df (pd.DataFrame): メタデータ(id, path, label, texture)を持つデータフレーム

    Returns:
        plt.Figure: 描画したfig
    """
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5 * 4, 5 * 4))
    axes = np.ravel(axes)
    sample_df = df.sample(min(4 * 4, len(df)), random_state=CFG.seed).reset_index(
        drop=True
    )
    for i in range(min(4 * 4, len(df))):
        img = Image.open(df["path"][i])
        id, label, texture = (
            sample_df["id"][i],
            sample_df["label"][i],
            sample_df["texture"][i],
        )
        axes[i].set_title(f"id={id}, label={label}, texture={texture}")
        axes[i].imshow(img)
    fig.tight_layout()
    return fig


@click.command()
@click.option("--config_path", "-c", default="../config/preprocess_cfg.yaml")
def main(config_path):
    # setup
    setup(config_path)

    # load data
    base_df = get_base_df()

    # preprocess
    img_hash = get_img_hash(base_df)
    texture_df = get_texture_df(base_df, img_hash)
    pseudo_df = add_pseudo_label(texture_df, img_hash)

    # train test split
    train_pseudo_df = pseudo_df.query("is_train == True").reset_index(drop=True)
    test_pseudo_df = pseudo_df.query("is_train == False").reset_index(drop=True)

    # save
    train_pseudo_df.to_csv(OUTPUT_DIR / "train_pseudo.csv", index=False)
    test_pseudo_df.to_csv(OUTPUT_DIR / "test_pseudo.csv", index=False)

    # テクスチャ毎に異常値のみを描画
    for texture in test_pseudo_df["texture"].unique():
        _df = test_pseudo_df.query("texture == @texture and label==1").reset_index(
            drop=True
        )
        fig = plot_pseudo_annomaly(_df)
        fig.savefig(str(OUTPUT_DIR / f"sample_pseudo_{texture}.jpg"))


if __name__ == "__main__":
    main()
