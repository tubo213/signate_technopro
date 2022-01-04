import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim  # noqa: F401
from models.augmentation import get_default_transforms, mixup
from models.loss import SCELoss  # noqa: F401
from pytorch_grad_cam import GradCAMPlusPlus
from sklearn.metrics import log_loss, roc_auc_score
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss.name)(**self.cfg.loss.params)
        self.transform = get_default_transforms()
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=1
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float()
        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()

        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()

        metrics = log_loss(labels, preds)
        auc = roc_auc_score(labels, preds)
        self.log(f"{mode}_loss", metrics, prog_bar=True)
        self.log(f"{mode}_auc", auc, prog_bar=True)

    def check_gradcam(
        self, dataloader, target_layers, target_category, reshape_transform=None
    ):
        cam = GradCAMPlusPlus(
            model=self,
            target_layers=target_layers,
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform,
        )

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform["val"](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy()
        labels = labels.cpu().numpy()

        grayscale_cam = cam(
            input_tensor=images, target_category=target_category, eigen_smooth=True
        )
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.0
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]


def inference(model: Model, dataloader: DataLoader) -> np.array:
    """推論

    Args:
        model (Model): モデル
        dataloader (DataLoader): 推論データローダー

    Returns:
        np.array: 推論結果
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    transform = get_default_transforms()
    preds = []

    for images in tqdm(dataloader, desc="[Inference]"):
        images = transform["val"](images).to(device)
        pred = model(images).sigmoid().to("cpu").detach().numpy().copy()
        preds.append(pred)

    preds = np.concatenate(preds).squeeze()
    return preds
