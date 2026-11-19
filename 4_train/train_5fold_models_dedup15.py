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
# Project configuration
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

PATCH_CSV = (
    ROOT /
    "2_dataset" /
    "finetune_patch_index_v2.csv"
)

FOLD_DIR = (
    ROOT /
    "2_dataset" /
    "folds_dedup15"
)

CKPT_DIR = (
    ROOT /
    "5_checkpoints" /
    "folds_dedup15"
)

LOG_DIR = (
    ROOT /
    "4_results" /
    "folds_dedup15"
)

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Training parameters
# ============================================================

EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
BASE_CH = 16
BASE_SEED = 2026


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

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(self, idx):

        row = self.df.iloc[
            idx
        ]

        image = np.load(
            ROOT /
            row["image_npy"]
        )

        label = np.load(
            ROOT /
            row["label_npy"]
        )

        z0 = int(
            row["z0"]
        )

        y0 = int(
            row["y0"]
        )

        x0 = int(
            row["x0"]
        )

        dz = int(
            row["dz"]
        )

        dy = int(
            row["dy"]
        )

        dx = int(
            row["dx"]
        )

        image = image[
            z0:z0 + dz,
            y0:y0 + dy,
            x0:x0 + dx
        ]

        label = label[
            z0:z0 + dz,
            y0:y0 + dy,
            x0:x0 + dx
        ]

        # ----------------------------------------------------
        # Safe padding
        # ----------------------------------------------------

        pad_z = max(
            0,
            dz - image.shape[0]
        )

        pad_y = max(
            0,
            dy - image.shape[1]
        )

        pad_x = max(
            0,
            dx - image.shape[2]
        )

        if (
            pad_z > 0
            or
            pad_y > 0
            or
            pad_x > 0
        ):

            image = np.pad(
                image,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant"
            )

            label = np.pad(
                label,
                (
                    (0, pad_z),
                    (0, pad_y),
                    (0, pad_x)
                ),
                mode="constant"
            )

        image = (
            torch
            .from_numpy(image)
            .float()
            .unsqueeze(0)
        )

        label = (
            torch
            .from_numpy(label)
            .long()
        )

        return image, label


# ============================================================
# Model builder
# ============================================================

def build_model(model_name):

    if model_name == "unet3d":

        model_module = SourceFileLoader(
            "unet3d_dedup15",
            str(
                ROOT /
                "3_models" /
                "unet3d.py"
            )
        ).load_module()

        model = model_module.UNet3D(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "attention_unet3d":

        model_module = SourceFileLoader(
            "attention_unet3d_dedup15",
            str(
                ROOT /
                "3_models" /
                "attention_unet3d.py"
            )
        ).load_module()

        model = model_module.AttentionUNet3D(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_no_multiscale":

        model_module = SourceFileLoader(
            "msau_no_multiscale_dedup15",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_no_multiscale.py"
            )
        ).load_module()

        model = model_module.MSAUNet3D_NoMultiScale(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_no_msa":

        model_module = SourceFileLoader(
            "msau_no_msa_dedup15",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_msa_only.py"
            )
        ).load_module()

        model = model_module.MSAUNet3DMSAOnly(
            in_channels=1,
            num_classes=2,
            base_ch=BASE_CH
        )

    elif model_name == "msau_unet3d_v2":

        model_module = SourceFileLoader(
            "msau_unet3d_v2_dedup15",
            str(
                ROOT /
                "3_models" /
                "msau_unet3d_v2.py"
            )
        ).load_module()

        model = model_module.MSAUNet3DV2(
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
        choices=[
            1,
            2,
            3,
            4,
            5
        ]
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "unet3d",
            "attention_unet3d",
            "msau_no_multiscale",
            "msau_no_msa",
            "msau_unet3d_v2"
        ]
    )

    args = parser.parse_args()

    fold = args.fold
    model_name = args.model

    run_seed = (
        BASE_SEED +
        fold
    )

    set_seed(
        run_seed
    )

    print("=" * 90)
    print("DEDUPLICATED 15-CASE STRICT FIVE-FOLD TRAINING")
    print("=" * 90)

    print(
        "Model      :",
        model_name
    )

    print(
        "Fold       :",
        fold
    )

    print(
        "Seed       :",
        run_seed
    )

    print(
        "Epochs     :",
        EPOCHS
    )

    print(
        "Batch size :",
        BATCH_SIZE
    )

    print(
        "Learning rate:",
        LR
    )


    # ========================================================
    # Fold files
    # ========================================================

    train_csv = (
        FOLD_DIR /
        f"fold{fold}_train.csv"
    )

    test_csv = (
        FOLD_DIR /
        f"fold{fold}_test.csv"
    )

    if not train_csv.exists():

        raise FileNotFoundError(
            train_csv
        )

    if not test_csv.exists():

        raise FileNotFoundError(
            test_csv
        )


    train_case_df = pd.read_csv(
        train_csv
    )

    test_case_df = pd.read_csv(
        test_csv
    )


    train_cases = (
        train_case_df[
            "case_id"
        ]
        .astype(str)
        .tolist()
    )

    test_cases = (
        test_case_df[
            "case_id"
        ]
        .astype(str)
        .tolist()
    )


    # ========================================================
    # Leakage check
    # ========================================================

    overlap = (
        set(train_cases)
        &
        set(test_cases)
    )

    if len(overlap) > 0:

        raise RuntimeError(
            "Train/test leakage detected: "
            +
            str(
                sorted(overlap)
            )
        )


    print(
        "\nTrain cases:",
        len(train_cases)
    )

    print(
        train_cases
    )

    print(
        "\nTest cases:",
        len(test_cases)
    )

    print(
        test_cases
    )

    print(
        "\nOverlap:",
        len(overlap)
    )


    # ========================================================
    # Patch selection
    # ========================================================

    patch_df = pd.read_csv(
        PATCH_CSV
    )

    patch_df[
        "case_id"
    ] = (
        patch_df[
            "case_id"
        ]
        .astype(str)
    )

    train_df = patch_df[
        patch_df[
            "case_id"
        ].isin(
            train_cases
        )
    ].copy()


    if len(train_df) == 0:

        raise RuntimeError(
            "No training patches found."
        )


    # --------------------------------------------------------
    # Additional patch leakage audit
    # --------------------------------------------------------

    patch_case_set = set(
        train_df[
            "case_id"
        ]
        .unique()
    )

    patch_overlap = (
        patch_case_set
        &
        set(test_cases)
    )

    if len(patch_overlap) > 0:

        raise RuntimeError(
            "Patch-level leakage detected: "
            +
            str(
                sorted(
                    patch_overlap
                )
            )
        )


    print(
        "Train patches:",
        len(train_df)
    )

    print(
        "Patch/test overlap:",
        len(
            patch_overlap
        )
    )


    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # ========================================================
    # DataLoader
    # ========================================================

    generator = torch.Generator()

    generator.manual_seed(
        run_seed
    )

    dataset = AortaPatchDataset(
        train_df
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )


    # ========================================================
    # Model
    # ========================================================

    model = build_model(
        model_name
    )

    model = model.to(
        device
    )


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

    loss_module = SourceFileLoader(
        "losses_dedup15",
        str(
            ROOT /
            "6_metrics" /
            "losses.py"
        )
    ).load_module()


    dice_loss = (
        loss_module.DiceLoss()
    )

    ce_loss = (
        torch.nn.CrossEntropyLoss()
    )


    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LR
        )
    )


    # ========================================================
    # Output files
    # ========================================================

    checkpoint_path = (
        CKPT_DIR /
        f"fold{fold}_{model_name}_best.pth"
    )

    training_log_path = (
        LOG_DIR /
        f"fold{fold}_{model_name}_training_log.csv"
    )


    # ========================================================
    # Training
    # ========================================================

    best_dice = -1.0
    best_epoch = -1

    log_rows = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        total_dice = 0.0

        batch_count = 0


        for image, label in dataloader:

            image = image.to(
                device,
                non_blocking=True
            )

            label = label.to(
                device,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = model(
                image
            )


            # ------------------------------------------------
            # Shape check
            # ------------------------------------------------

            if (
                logits.shape[2:]
                !=
                label.shape[1:]
            ):

                raise RuntimeError(
                    "Output/label shape mismatch: "
                    f"logits={tuple(logits.shape)}, "
                    f"label={tuple(label.shape)}"
                )


            loss = (
                ce_loss(
                    logits,
                    label
                )
                +
                dice_loss(
                    logits,
                    label
                )
            )


            loss.backward()

            optimizer.step()


            dice = (
                loss_module.dice_score(
                    logits.detach(),
                    label.detach()
                )
            )


            if torch.is_tensor(
                dice
            ):

                dice = (
                    dice
                    .detach()
                    .cpu()
                    .item()
                )


            total_loss += float(
                loss.item()
            )

            total_dice += float(
                dice
            )

            batch_count += 1


        epoch_loss = (
            total_loss /
            batch_count
        )

        epoch_dice = (
            total_dice /
            batch_count
        )


        print(
            f"Model {model_name:22s} "
            f"Fold {fold} "
            f"Epoch {epoch:03d}/{EPOCHS:03d} "
            f"Loss {epoch_loss:.4f} "
            f"Dice {epoch_dice:.4f}"
        )


        log_rows.append(
            {
                "model":
                    model_name,

                "fold":
                    fold,

                "epoch":
                    epoch,

                "seed":
                    run_seed,

                "train_cases":
                    len(train_cases),

                "test_cases":
                    len(test_cases),

                "train_patches":
                    len(train_df),

                "loss":
                    epoch_loss,

                "dice":
                    epoch_dice
            }
        )


        # ----------------------------------------------------
        # Preserve same checkpoint criterion as previous runs
        # ----------------------------------------------------

        if epoch_dice > best_dice:

            best_dice = epoch_dice
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "Saved best:",
                checkpoint_path
            )


    # ========================================================
    # Save log
    # ========================================================

    pd.DataFrame(
        log_rows
    ).to_csv(
        training_log_path,
        index=False
    )


    print("\n" + "=" * 90)
    print("TRAINING FINISHED")
    print("=" * 90)

    print(
        "Model      :",
        model_name
    )

    print(
        "Fold       :",
        fold
    )

    print(
        "Best epoch :",
        best_epoch
    )

    print(
        "Best Dice  :",
        f"{best_dice:.6f}"
    )

    print(
        "Checkpoint :",
        checkpoint_path
    )

    print(
        "Log        :",
        training_log_path
    )


if __name__ == "__main__":
    main()
