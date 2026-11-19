import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

from importlib.machinery import SourceFileLoader

unet_mod = SourceFileLoader(
    "unet3d",
    str(ROOT / "3_models" / "msau_unet3d_msa_only.py")
).load_module()

loss_mod = SourceFileLoader(
    "losses",
    str(ROOT / "6_metrics" / "losses.py")
).load_module()


PATCH_CSV = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"

EPOCHS = 50
BATCH_SIZE = 1
LR = 1e-4


class AortaPatchDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img = np.load(ROOT / r["image_npy"])
        lab = np.load(ROOT / r["label_npy"])

        z0, y0, x0 = int(r["z0"]), int(r["y0"]), int(r["x0"])
        dz, dy, dx = int(r["dz"]), int(r["dy"]), int(r["dx"])

        img = img[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        lab = lab[z0:z0+dz, y0:y0+dy, x0:x0+dx]

        pad_z = max(0, dz - img.shape[0])
        pad_y = max(0, dy - img.shape[1])
        pad_x = max(0, dx - img.shape[2])

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            img = np.pad(
                img,
                ((0,pad_z),(0,pad_y),(0,pad_x)),
                mode="constant"
            )
            lab = np.pad(
                lab,
                ((0,pad_z),(0,pad_y),(0,pad_x)),
                mode="constant"
            )

        img = torch.from_numpy(img).float().unsqueeze(0)
        lab = torch.from_numpy(lab).long()

        return img, lab


def main():

    device = torch.device("cuda")

    ds = AortaPatchDataset(PATCH_CSV)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = unet_mod.MSAUNet3DMSAOnly(
        in_channels=1,
        num_classes=2,
        base_ch=16
    ).to(device)

    dice_loss = loss_mod.DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_dice = 0

    log_rows = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        epoch_loss = 0
        epoch_dice = 0

        for img, lab in dl:

            img = img.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(img)

            loss = (
                ce_loss(logits, lab)
                + dice_loss(logits, lab)
            )

            loss.backward()

            optimizer.step()

            dice = loss_mod.dice_score(
                logits.detach(),
                lab.detach()
            )

            epoch_loss += loss.item()
            epoch_dice += dice

        epoch_loss /= len(dl)
        epoch_dice /= len(dl)

        print(
            f"Epoch {epoch:03d} "
            f"Loss {epoch_loss:.4f} "
            f"Dice {epoch_dice:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "dice": epoch_dice
        })

        if epoch_dice > best_dice:
            best_dice = epoch_dice

            ckpt = ROOT / "5_checkpoints" / "finetune" / "msau_msa_only_best.pth"

            torch.save(model.state_dict(), ckpt)

            print("Saved best:", ckpt)

    log_df = pd.DataFrame(log_rows)

    out_csv = ROOT / "4_results" / "finetune" / "msau_msa_only_training_log.csv"

    log_df.to_csv(out_csv, index=False)

    print("\nTraining Finished")
    print("Best Dice:", best_dice)
    print("Log saved:", out_csv)


if __name__ == "__main__":
    main()
