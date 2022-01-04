import torch
import torch.nn as nn


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1, beta=0.5):
        super(SCELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.bce(pred, labels)

        # RCE
        pred = pred.sigmoid()
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = labels.float().to(self.device)
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        rce = -1 * (pred * torch.log(labels))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
