import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

from importlib.machinery import SourceFileLoader

unet_mod = SourceFileLoader("unet3d", str(ROOT / "3_models" / "unet3d.py")).load_module()
loss_mod = SourceFileLoader("losses", str(ROOT / "6_metrics" / "losses.py")).load_module()


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

        # pad to fixed patch size 96×96×96
        pad_z = max(0, dz - img.shape[0])
        pad_y = max(0, dy - img.shape[1])
        pad_x = max(0, dx - img.shape[2])

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            img = np.pad(
                img,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=0
            )
            lab = np.pad(
                lab,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=0
            )

        img = torch.from_numpy(img).float().unsqueeze(0)
        lab = torch.from_numpy(lab).long()

        return img, lab


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    csv_path = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"
    ds = AortaPatchDataset(csv_path)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = unet_mod.UNet3D(in_channels=1, num_classes=2, base_ch=8).to(device)

    dice_loss = loss_mod.DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_dice = 0

        for step, (img, lab) in enumerate(dl, start=1):
            img = img.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()

            logits = model(img)

            loss = ce_loss(logits, lab) + dice_loss(logits, lab)

            loss.backward()
            optimizer.step()

            dice = loss_mod.dice_score(logits.detach(), lab.detach())

            total_loss += loss.item()
            total_dice += dice

            if step % 20 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(dl)} "
                    f"Loss {total_loss/step:.4f} Dice {total_dice/step:.4f}"
                )

        print(
            f"Epoch {epoch} Finished "
            f"Loss {total_loss/len(dl):.4f} Dice {total_dice/len(dl):.4f}"
        )

    ckpt = ROOT / "5_checkpoints" / "finetune" / "unet3d_baseline_test.pth"
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
