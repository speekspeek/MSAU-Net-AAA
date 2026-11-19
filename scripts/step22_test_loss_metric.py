import sys
from pathlib import Path
import torch

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

from importlib.machinery import SourceFileLoader

losses = SourceFileLoader("losses", str(ROOT / "6_metrics" / "losses.py")).load_module()

logits = torch.randn(2, 2, 96, 96, 96)
targets = torch.randint(0, 2, (2, 96, 96, 96))

criterion = losses.DiceLoss()
loss = criterion(logits, targets)
dice = losses.dice_score(logits, targets)

print("Loss:", float(loss))
print("Dice:", dice)
