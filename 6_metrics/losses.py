import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        targets = (targets == 1).float()

        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        denominator = torch.sum(probs, dims) + torch.sum(targets, dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def dice_score(logits, targets, threshold=0.5, smooth=1e-5):
    probs = F.softmax(logits, dim=1)[:, 1]
    preds = (probs > threshold).float()
    targets = (targets == 1).float()

    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    denominator = torch.sum(preds, dims) + torch.sum(targets, dims)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean().item()
