import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from models.dataset import CustomDataset
from models.model import Model, inference
from torch.utils.data import DataLoader
from utils import load_config


def load_test_df(config):
    input_dir = Path(config.input_dir)

    test_df = pd.DataFrame(
        list(map(lambda x: x.stem, input_dir.glob("test/*"))), columns=["id"]
    )
    test_df["path"] = test_df["id"].apply(
        lambda x: str(input_dir / "test" / x) + ".png"
    )
    return test_df


@click.command()
@click.option(
    "--weight_dir",
    "-d",
    default="../output/exp_001/swin_tiny_patch4_window7_224/thresh85/default/",
)
@click.option("--config_path", "-c", default="../config/train_cfg.yaml")
def main(weight_dir, config_path):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    weight_dir = Path(weight_dir)
    config = load_config(config_path)
    output_dir = Path(f"../output/{config.exp_name}/{config.model.name}/")

    test_df = load_test_df(config)
    test_preds = np.zeros(len(test_df))

    for fold in range(config.n_splits):
        test_dataset = CustomDataset(
            test_df[["path"]], image_size=config.transform.image_size
        )
        test_dataloader = DataLoader(test_dataset, **config.val_loader)
        weight_path = weight_dir / f"version_{fold}/checkpoints/best_loss.ckpt"

        # load weight
        model = Model(config)
        model.load_state_dict(torch.load(weight_path)["state_dict"])

        # inference
        test_preds += inference(model, test_dataloader)

    test_preds /= config.n_splits
    test_df["label"] = test_preds
    test_df[["id", "label"]].to_csv(
        output_dir / "submission.tsv", index=False, header=None, sep="\t"
    )


if __name__ == "__main__":
    main()
