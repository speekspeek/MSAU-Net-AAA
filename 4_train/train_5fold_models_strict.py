import sys
import argparse
import random
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

PATCH_CSV = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"
FOLD_DIR = ROOT / "2_dataset" / "folds"

CKPT_DIR = ROOT / "5_checkpoints" / "folds"
LOG_DIR = ROOT / "4_results" / "folds"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
BASE_CH = 16
SEED = 2026


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class AortaPatchDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        r = self.df.iloc[idx]

        img = np.load(ROOT / r["image_npy"])
        lab = np.load(ROOT / r["label_npy"])

        z0 = int(r["z0"])
        y0 = int(r["y0"])
        x0 = int(r["x0"])

        dz = int(r["dz"])
        dy = int(r["dy"])
        dx = int(r["dx"])

        img = img[
            z0:z0 + dz,
            y0:y0 + dy,
            x0:x0 + dx
        ]

        lab = lab[
            z0:z0 + dz,
            y0:y0 + dy,
            x0:x0 + dx
        ]

        pad_z = max(0, dz - img.shape[0])
        pad_y = max(0, dy - img.shape[1])
        pad_x = max(0, dx - img.shape[2])

        if pad_z or pad_y or pad_x:

            img = np.pad(
                img,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant"
            )

            lab = np.pad(
                lab,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant"
            )

        img = torch.from_numpy(img).float().unsqueeze(0)
        lab = torch.from_numpy(lab).long()

        return img, lab


# ============================================================
# Model builder
# ============================================================

def build_model(model_name):

    if model_name == "unet3d":

        mod = SourceFileLoader(
            "unet3d_strict",
            str(ROOT / "3_models" / "unet3d.py")
        ).load_module()

        model = mod.UNet3D(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_no_multiscale":

        mod = SourceFileLoader(
            "msau_no_multiscale_strict",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_no_multiscale.py"
            )
        ).load_module()

        model = mod.MSAUNet3D_NoMultiScale(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_no_msa":

        mod = SourceFileLoader(
            "msau_no_msa_strict",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_msa_only.py"
            )
        ).load_module()

        model = mod.MSAUNet3DMSAOnly(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_unet3d_v2":

        mod = SourceFileLoader(
            "msau_unet3d_v2_strict",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_v2.py"
            )
        ).load_module()

        model = mod.MSAUNet3DV2(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5]
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "unet3d",
            "msau_no_multiscale",
            "msau_no_msa",
            "msau_unet3d_v2"
        ]
    )

    args = parser.parse_args()

    fold = args.fold
    model_name = args.model

    run_seed = SEED + fold
    set_seed(run_seed)

    print("=" * 80)
    print("STRICT FIVE-FOLD TRAINING")
    print("=" * 80)

    print("Model      :", model_name)
    print("Fold       :", fold)
    print("Seed       :", run_seed)
    print("Epochs     :", EPOCHS)
    print("Batch size :", BATCH_SIZE)
    print("LR         :", LR)

    # ========================================================
    # Read fold
    # ========================================================

    train_csv = FOLD_DIR / f"fold{fold}_train.csv"
    test_csv = FOLD_DIR / f"fold{fold}_test.csv"

    train_cases = (
        pd.read_csv(train_csv)["case_id"]
        .astype(str)
        .tolist()
    )

    test_cases = (
        pd.read_csv(test_csv)["case_id"]
        .astype(str)
        .tolist()
    )

    overlap = set(train_cases) & set(test_cases)

    if overlap:
        raise RuntimeError(
            f"Train/test leakage detected: {sorted(overlap)}"
        )

    print("\nTrain cases:", len(train_cases))
    print("Test cases :", len(test_cases))
    print("Overlap    :", len(overlap))

    # ========================================================
    # Training patches
    # ========================================================

    patch_df = pd.read_csv(PATCH_CSV)

    patch_df["case_id"] = (
        patch_df["case_id"].astype(str)
    )

    train_df = patch_df[
        patch_df["case_id"].isin(train_cases)
    ].copy()

    if len(train_df) == 0:
        raise RuntimeError(
            "No training patches found."
        )

    print("Train patches:", len(train_df))

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # DataLoader
    # ========================================================

    generator = torch.Generator()
    generator.manual_seed(run_seed)

    ds = AortaPatchDataset(train_df)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )

    # ========================================================
    # Model
    # ========================================================

    model = build_model(model_name).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{total_params:,}"
    )

    # ========================================================
    # Loss
    # ========================================================

    loss_mod = SourceFileLoader(
        "losses_strict",
        str(ROOT / "6_metrics" / "losses.py")
    ).load_module()

    dice_loss = loss_mod.DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # ========================================================
    # Output
    # ========================================================

    ckpt_path = (
        CKPT_DIR /
        f"fold{fold}_{model_name}_best.pth"
    )

    log_path = (
        LOG_DIR /
        f"fold{fold}_{model_name}_training_log.csv"
    )

    # ========================================================
    # Training
    # ========================================================

    best_dice = 0.0
    best_epoch = 0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        epoch_loss = 0.0
        epoch_dice = 0.0

        for img, lab in dl:

            img = img.to(
                device,
                non_blocking=True
            )

            lab = lab.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            logits = model(img)

            if logits.shape[2:] != lab.shape[1:]:

                raise RuntimeError(
                    f"Output/label mismatch: "
                    f"{tuple(logits.shape)} vs "
                    f"{tuple(lab.shape)}"
                )

            loss = (
                ce_loss(logits, lab)
                +
                dice_loss(logits, lab)
            )

            loss.backward()
            optimizer.step()

            dice = loss_mod.dice_score(
                logits.detach(),
                lab.detach()
            )

            if torch.is_tensor(dice):
                dice = dice.item()

            epoch_loss += loss.item()
            epoch_dice += float(dice)

        epoch_loss /= len(dl)
        epoch_dice /= len(dl)

        print(
            f"Model {model_name:22s} "
            f"Fold {fold} "
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"Loss {epoch_loss:.4f} "
            f"Dice {epoch_dice:.4f}"
        )

        log_rows.append({
            "model": model_name,
            "fold": fold,
            "epoch": epoch,
            "loss": epoch_loss,
            "dice": epoch_dice
        })

        if epoch_dice > best_dice:

            best_dice = epoch_dice
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                ckpt_path
            )

            print(
                "Saved best:",
                ckpt_path
            )

    # ========================================================
    # Save log
    # ========================================================

    pd.DataFrame(
        log_rows
    ).to_csv(
        log_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print("Model      :", model_name)
    print("Fold       :", fold)
    print("Best epoch :", best_epoch)
    print("Best Dice  :", f"{best_dice:.6f}")
    print("Checkpoint :", ckpt_path)
    print("Log        :", log_path)


if __name__ == "__main__":
    main()
