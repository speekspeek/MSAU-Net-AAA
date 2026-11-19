import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

sys.path.append(str(ROOT))


from importlib.machinery import SourceFileLoader


model_mod = SourceFileLoader(
    "model",
    str(ROOT / "3_models" / "msau_unet3d_no_multiscale.py")
).load_module()


loss_mod = SourceFileLoader(
    "loss",
    str(ROOT / "6_metrics" / "losses.py")
).load_module()



PATCH_CSV = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"


EPOCHS = 50
BATCH_SIZE = 1
LR = 1e-4




class Dataset3D(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        r = self.df.iloc[idx]


        img = np.load(
            ROOT / r["image_npy"]
        )


        lab = np.load(
            ROOT / r["label_npy"]
        )


        img = torch.from_numpy(
            img
        ).float().unsqueeze(0)


        lab = torch.from_numpy(
            lab
        ).long()


        return img, lab





def resize_label(label, target_size):

    """
    保证label与prediction尺寸一致
    """

    if label.shape[1:] == target_size:

        return label


    label = F.interpolate(
        label.unsqueeze(1).float(),
        size=target_size,
        mode="nearest"
    )


    return label.squeeze(1).long()





def main():


    device = torch.device("cuda")


    dataset = Dataset3D(
        PATCH_CSV
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )



    model = model_mod.MSAUNet3D_NoMultiScale(
        in_channels=1,
        num_classes=2,
        base_ch=16
    ).to(device)



    ce_loss = torch.nn.CrossEntropyLoss()


    dice_loss = loss_mod.DiceLoss()



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )



    best_dice = 0



    log = []



    for epoch in range(1, EPOCHS+1):


        model.train()


        total_loss = 0

        total_dice = 0



        for img, lab in loader:


            img = img.to(
                device,
                non_blocking=True
            )


            lab = lab.to(
                device,
                non_blocking=True
            )



            optimizer.zero_grad()



            pred = model(img)



            # ===========================
            # 尺寸自动对齐
            # ===========================

            lab = resize_label(
                lab,
                pred.shape[2:]
            )



            loss = (
                ce_loss(pred, lab)
                +
                dice_loss(pred, lab)
            )



            loss.backward()


            optimizer.step()



            dice = loss_mod.dice_score(
                pred.detach(),
                lab.detach()
            )



            total_loss += loss.item()

            total_dice += dice



        epoch_loss = total_loss / len(loader)

        epoch_dice = total_dice / len(loader)



        print(
            f"Epoch {epoch:03d} "
            f"Loss {epoch_loss:.4f} "
            f"Dice {epoch_dice:.4f}"
        )



        log.append(
            {
                "epoch":epoch,
                "loss":epoch_loss,
                "dice":epoch_dice
            }
        )



        if epoch_dice > best_dice:


            best_dice = epoch_dice


            ckpt = (
                ROOT
                /
                "5_checkpoints"
                /
                "finetune"
                /
                "msau_no_multiscale_best.pth"
            )


            torch.save(
                model.state_dict(),
                ckpt
            )


            print(
                "Saved best:",
                ckpt
            )




    out_csv = (
        ROOT
        /
        "4_results"
        /
        "finetune"
        /
        "msau_no_multiscale_training_log.csv"
    )


    pd.DataFrame(log).to_csv(
        out_csv,
        index=False
    )


    print("\nTraining Finished")

    print(
        "Best Dice:",
        best_dice
    )

    print(
        "Log saved:",
        out_csv
    )





if __name__ == "__main__":

    main()
