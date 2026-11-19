#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import (
    zoom,
    label as cc_label,
    binary_erosion,
    distance_transform_edt
)


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path.home() / "AAA_CTA_Project"

MANIFEST = ROOT / (
    "2_dataset/"
    "finetune_preprocessed_manifest_v3_corrected_dedup15.csv"
)

CORRECTED = ROOT / "0_raw_data_corrected"

OUT = ROOT / "9_results/auto_roi_5fold_v1"

MODEL_DIR = OUT / "locator_checkpoints"

TARGET_SPACING = (1.5, 1.5, 1.5)

COARSE_SHAPE = (64, 64, 128)

ROI_MARGIN = 24

FOLDS = {
    1: ("Anev_04", "Anev_09", "Nat_04"),
    2: ("Anev_06", "Nat_01", "Nat_03"),
    3: ("Anev_01", "Nat_07", "Nat_09"),
    4: ("Anev_02", "Nat_02", "Nat_06"),
    5: ("Anev_03", "Nat_05", "Nat_08"),
}


# ============================================================
# 2. Reproducibility
# ============================================================

def seed_all(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Verify independent cases
# ============================================================

def verify_cases():

    df = pd.read_csv(MANIFEST)

    cases = df["case_id"].astype(str).tolist()

    assert len(cases) == len(set(cases)) == 15, (
        "Expected 15 unique cases"
    )

    tests = [
        case
        for group in FOLDS.values()
        for case in group
    ]

    assert sorted(cases) == sorted(tests), (
        "Manifest and fixed OOF folds differ"
    )

    return cases


# ============================================================
# 4. Image and GT paths
# ============================================================

def paths(case):

    group = (
        "Anev"
        if case.startswith("Anev")
        else "Nat"
    )

    image = (
        ROOT /
        "0_raw_data" /
        group /
        case /
        "nifti" /
        f"{case}_image.nii.gz"
    )

    matches = sorted(
        CORRECTED.rglob(
            f"{case}_segmentation_corrected.nii.gz"
        )
    )

    if not image.is_file() or len(matches) != 1:
        raise FileNotFoundError(
            f"{case}: image={image.exists()} "
            f"corrected_labels={matches}"
        )

    return image, matches[0]


# ============================================================
# 5. Load and resample volume
# ============================================================

def load_volume(case, need_gt):

    image_path, gt_path = paths(case)

    nii = nib.load(str(image_path))

    spacing = nii.header.get_zooms()[:3]

    factors = tuple(
        float(spacing[i]) / TARGET_SPACING[i]
        for i in range(3)
    )

    image = nii.get_fdata(dtype=np.float32)

    image = zoom(
        image,
        factors,
        order=1
    ).astype(np.float32)

    np.clip(
        image,
        -200.0,
        400.0,
        out=image
    )

    gt = None

    if need_gt:

        gt_nii = nib.load(str(gt_path))

        if (
            gt_nii.shape != nii.shape
            or not np.allclose(
                gt_nii.affine,
                nii.affine,
                atol=1e-4
            )
        ):
            raise RuntimeError(
                f"{case}: image/GT physical-space mismatch"
            )

        raw_gt = (
            gt_nii.get_fdata(dtype=np.float32) > 0
        ).astype(np.uint8)

        gt = zoom(
            raw_gt,
            factors,
            order=0
        ).astype(bool)

        if gt.shape != image.shape:
            raise RuntimeError(
                f"{case}: resampled image/GT mismatch"
            )

    return image, gt, nii.affine, factors


# ============================================================
# 6. Coarse-resolution image
# ============================================================

def coarse_image(image):

    factors = tuple(
        COARSE_SHAPE[i] / image.shape[i]
        for i in range(3)
    )

    small = zoom(
        image,
        factors,
        order=1
    ).astype(np.float32)

    if small.shape != COARSE_SHAPE:
        raise RuntimeError(
            f"Unexpected coarse image shape: {small.shape}"
        )

    mean = float(small.mean())
    std = float(small.std())

    small = (
        small - mean
    ) / (
        std + 1e-8
    )

    return small


def coarse_gt(gt):

    factors = tuple(
        COARSE_SHAPE[i] / gt.shape[i]
        for i in range(3)
    )

    small = zoom(
        gt.astype(np.uint8),
        factors,
        order=0
    ).astype(np.float32)

    if small.shape != COARSE_SHAPE:
        raise RuntimeError(
            f"Unexpected coarse GT shape: {small.shape}"
        )

    return small


# ============================================================
# 7. Lightweight 3D U-Net
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, cin, cout):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(
                cin, cout,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                cout,
                affine=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv3d(
                cout, cout,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                cout,
                affine=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            )
        )

    def forward(self, x):
        return self.net(x)


class LocatorUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = ConvBlock(1, 8)
        self.e2 = ConvBlock(8, 16)
        self.e3 = ConvBlock(16, 32)

        self.b = ConvBlock(32, 48)

        self.d3 = ConvBlock(48 + 32, 32)
        self.d2 = ConvBlock(32 + 16, 16)
        self.d1 = ConvBlock(16 + 8, 8)

        self.head = nn.Conv3d(8, 1, 1)

    def forward(self, x):

        a = self.e1(x)

        b = self.e2(
            F.avg_pool3d(a, 2)
        )

        c = self.e3(
            F.avg_pool3d(b, 2)
        )

        z = self.b(
            F.avg_pool3d(c, 2)
        )

        z = self.d3(
            torch.cat([
                F.interpolate(
                    z,
                    size=c.shape[2:],
                    mode="trilinear",
                    align_corners=False
                ),
                c
            ], dim=1)
        )

        z = self.d2(
            torch.cat([
                F.interpolate(
                    z,
                    size=b.shape[2:],
                    mode="trilinear",
                    align_corners=False
                ),
                b
            ], dim=1)
        )

        z = self.d1(
            torch.cat([
                F.interpolate(
                    z,
                    size=a.shape[2:],
                    mode="trilinear",
                    align_corners=False
                ),
                a
            ], dim=1)
        )

        return self.head(z)


# ============================================================
# 8. BCE + Dice loss
# ============================================================

def loss_fn(logits, target):

    prob = torch.sigmoid(logits)

    dice = (
        2 * (prob * target).sum() + 1
    ) / (
        prob.sum() + target.sum() + 1
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target
    )

    return 0.5 * bce + 0.5 * (1 - dice)


# ============================================================
# 9. Internal train-validation split
# ============================================================

def split_train_val(fold, cases):

    outer = set(FOLDS[fold])

    pool = sorted(
        set(cases) - outer
    )

    rng = random.Random(
        1700 + fold
    )

    shuffled = pool[:]
    rng.shuffle(shuffled)

    val = sorted(shuffled[:2])
    train = sorted(shuffled[2:])

    assert len(train) == 10
    assert len(val) == 2

    return train, val


# ============================================================
# 10. Train one fold
# ============================================================

def train_fold(fold, epochs, lr):

    cases = verify_cases()

    train, val = split_train_val(
        fold,
        cases
    )

    seed_all(2026 + fold)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Fold {fold} "
        f"train={train} "
        f"val={val} "
        f"outer_test={FOLDS[fold]}",
        flush=True
    )

    # Outer-test images and GT are not loaded.

    tensors = {}

    for case in train + val:

        img, gt, _, _ = load_volume(
            case,
            need_gt=True
        )

        x = torch.from_numpy(
            coarse_image(img)[None, None].copy()
        )

        y = torch.from_numpy(
            coarse_gt(gt)[None, None].copy()
        )

        tensors[case] = (x, y)

        del img, gt

    model = LocatorUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best = -1.0

    history = []

    checkpoint = (
        MODEL_DIR /
        f"fold{fold}_locator_best.pth"
    )

    for epoch in range(1, epochs + 1):

        model.train()

        random.shuffle(train)

        losses = []

        for case in train:

            x, y = tensors[case]

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        # Internal validation

        model.eval()

        val_dice = []

        with torch.no_grad():

            for case in val:

                x, y = tensors[case]

                pred = (
                    torch.sigmoid(
                        model(x.to(device))
                    ) >= 0.5
                )

                yy = y.to(device) > 0.5

                intersection = (
                    pred & yy
                ).sum().item()

                denominator = (
                    pred.sum().item()
                    +
                    yy.sum().item()
                )

                d = (
                    2 * intersection + 1
                ) / (
                    denominator + 1
                )

                val_dice.append(d)

        score = float(
            np.mean(val_dice)
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_coarse_dice": score
        })

        if score > best:

            best = score

            torch.save(
                model.state_dict(),
                checkpoint
            )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Fold={fold} "
                f"epoch={epoch:03d} "
                f"loss={np.mean(losses):.4f} "
                f"val_dice={score:.4f} "
                f"best={best:.4f}",
                flush=True
            )

    pd.DataFrame(history).to_csv(
        MODEL_DIR /
        f"fold{fold}_training_history.csv",
        index=False
    )

    print(
        "Saved:",
        checkpoint,
        "best_val_coarse_dice:",
        best
    )


# ============================================================
# 11. Automatic ROI localization
# ============================================================

def roi_from_locator(image, model, device):

    # Image-only inference. No reference GT is used.

    x = torch.from_numpy(
        coarse_image(image)[None, None].copy()
    ).to(device)

    model.eval()

    with torch.no_grad():

        probability = (
            torch.sigmoid(model(x))[0, 0]
            .cpu()
            .numpy()
        )

    coarse_mask = probability >= 0.5

    components, n = cc_label(
        coarse_mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=np.uint8
        )
    )

    if n > 0:

        sizes = np.bincount(
            components.ravel()
        )

        sizes[0] = 0

        largest_id = np.argmax(sizes)

        coarse_mask = (
            components == largest_id
        )

    # Empty prediction fallback:
    # use full CTA, not GT-guided cropping.

    if not coarse_mask.any():

        low = np.zeros(
            3,
            dtype=int
        )

        high = np.array(
            image.shape,
            dtype=int
        )

        return (
            low,
            high,
            "empty_locator_full_volume"
        )

    coords = np.argwhere(
        coarse_mask
    )

    low_coarse = coords.min(axis=0)

    high_coarse = (
        coords.max(axis=0) + 1
    )

    shape = np.array(
        image.shape
    )

    coarse_shape = np.array(
        COARSE_SHAPE
    )

    low = np.floor(
        low_coarse * shape / coarse_shape
    ).astype(int)

    high = np.ceil(
        high_coarse * shape / coarse_shape
    ).astype(int)

    low = np.maximum(
        low - ROI_MARGIN,
        0
    )

    high = np.minimum(
        high + ROI_MARGIN,
        shape
    )

    return (
        low,
        high,
        "locator_largest_component"
    )


# ============================================================
# 12. Segmentation metrics
# ============================================================

def metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    p = int(pred.sum())
    g = int(gt.sum())

    inter = int(
        (pred & gt).sum()
    )

    union = p + g - inter

    result = {
        "Dice": (
            2 * inter / (p + g)
            if p + g else 1.0
        ),

        "IoU": (
            inter / union
            if union else 1.0
        ),

        "Precision": (
            inter / p
            if p else 0.0
        ),

        "Recall": (
            inter / g
            if g else 0.0
        ),

        "Pred_voxels": p,
        "GT_voxels": g
    }

    if not p or not g:

        result.update({
            "HD95_mm": float("nan"),
            "ASSD_mm": float("nan")
        })

        return result

    structure = np.ones(
        (3, 3, 3),
        dtype=bool
    )

    pred_surface = (
        pred &
        ~binary_erosion(
            pred,
            structure=structure
        )
    )

    gt_surface = (
        gt &
        ~binary_erosion(
            gt,
            structure=structure
        )
    )

    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=TARGET_SPACING
    )

    d_pred_gt = dt_gt[
        pred_surface
    ]

    del dt_gt

    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=TARGET_SPACING
    )

    d_gt_pred = dt_pred[
        gt_surface
    ]

    del dt_pred

    distances = np.concatenate([
        d_pred_gt,
        d_gt_pred
    ])

    result["HD95_mm"] = float(
        np.percentile(distances, 95)
    )

    result["ASSD_mm"] = float(
        distances.mean()
    )

    return result


# ============================================================
# 13. Evaluate one outer fold
# ============================================================

def evaluate_fold(fold):

    verify_cases()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loc_ckpt = (
        MODEL_DIR /
        f"fold{fold}_locator_best.pth"
    )

    fine_ckpt = (
        ROOT /
        "5_checkpoints/folds_corrected_v5" /
        f"fold{fold}_msau_unet3d_v2_best.pth"
    )

    if (
        not loc_ckpt.exists()
        or not fine_ckpt.exists()
    ):
        raise FileNotFoundError(
            f"Missing checkpoints: "
            f"{loc_ckpt}, {fine_ckpt}"
        )

    # Load coarse locator

    locator = LocatorUNet().to(device)

    locator.load_state_dict(
        torch.load(
            loc_ckpt,
            map_location="cpu",
            weights_only=True
        )
    )

    locator.eval()

    # Load frozen Full MSAU-Net

    sys.path.insert(
        0,
        str(ROOT / "4_train")
    )

    train_mod = importlib.import_module(
        "train_corrected_v3_fold1_sanity"
    )

    fine = train_mod.build_model(
        "msau_unet3d_v2"
    )

    fine.load_state_dict(
        torch.load(
            fine_ckpt,
            map_location="cpu",
            weights_only=True
        )
    )

    fine = fine.to(device)
    fine.eval()

    fold_out = (
        OUT /
        f"fold{fold}"
    )

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for case in FOLDS[fold]:

        print(
            "Evaluating:",
            case,
            flush=True
        )

        start_time = time.perf_counter()

        # No GT is loaded for ROI or inference.

        image, _, affine, factors = load_volume(
            case,
            need_gt=False
        )

        low, high, status = roi_from_locator(
            image,
            locator,
            device
        )

        crop = image[
            low[0]:high[0],
            low[1]:high[1],
            low[2]:high[2]
        ].copy()

        crop_mean = float(crop.mean())
        crop_std = float(crop.std())

        crop = (
            (crop - crop_mean)
            /
            (crop_std + 1e-8)
        ).astype(np.float32)

        pred_crop = (
            train_mod.sliding_window_predict(
                fine,
                crop,
                device
            )
        )

        pred = np.zeros(
            image.shape,
            dtype=np.uint8
        )

        pred[
            low[0]:high[0],
            low[1]:high[1],
            low[2]:high[2]
        ] = pred_crop.astype(np.uint8)

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_seconds = (
            time.perf_counter() - start_time
        )

        # GT is loaded only after inference.

        _, gt, _, _ = load_volume(
            case,
            need_gt=True
        )

        result = metrics(
            pred,
            gt
        )

        row = {
            "case_id": case,
            "fold": fold,
            "roi_status": status,
            "roi_start": str(low.tolist()),
            "roi_stop_exclusive": str(high.tolist()),
            "roi_fraction": float(
                np.prod(
                    (high - low)
                    /
                    np.array(image.shape)
                )
            ),
            "inference_seconds": inference_seconds,
            **result
        }

        rows.append(row)

        rescale_affine = affine @ np.diag([
            1 / factors[0],
            1 / factors[1],
            1 / factors[2],
            1
        ])

        pred_nii = nib.Nifti1Image(
            pred,
            rescale_affine
        )

        nib.save(
            pred_nii,
            str(
                fold_out /
                f"{case}_prediction.nii.gz"
            )
        )

        print(
            f"{case} "
            f"Dice={result['Dice']:.4f} "
            f"Recall={result['Recall']:.4f} "
            f"HD95={result['HD95_mm']:.2f} "
            f"ROI={status}",
            flush=True
        )

        pd.DataFrame(rows).to_csv(
            fold_out / "metrics.csv",
            index=False
        )

        del image, gt, pred, crop, pred_crop

    print(
        "Saved:",
        fold_out / "metrics.csv"
    )


# ============================================================
# 14. Summarize all 15 OOF predictions
# ============================================================

def summarize():

    verify_cases()

    files = [
        OUT / f"fold{i}" / "metrics.csv"
        for i in FOLDS
    ]

    missing = [
        str(p)
        for p in files
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Run all folds first: {missing}"
        )

    df = pd.concat(
        [
            pd.read_csv(p)
            for p in files
        ],
        ignore_index=True
    )

    assert (
        len(df)
        ==
        df["case_id"].nunique()
        ==
        15
    ), "OOF count/uniqueness failure"

    for fold in FOLDS:

        actual = set(
            df.loc[
                df["fold"] == fold,
                "case_id"
            ]
        )

        expected = set(
            FOLDS[fold]
        )

        assert actual == expected, (
            f"Fold {fold}: OOF cases mismatch"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT / "all_15_oof_metrics.csv",
        index=False
    )

    columns = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "HD95_mm",
        "ASSD_mm"
    ]

    summary = (
        df[columns]
        .agg(["mean", "std"])
        .T
    )

    summary["valid_n"] = (
        df[columns].count()
    )

    summary.to_csv(
        OUT / "summary.csv"
    )

    print(
        summary.to_string()
    )

    print(
        "Saved:",
        OUT / "all_15_oof_metrics.csv",
        OUT / "summary.csv"
    )


# ============================================================
# 15. Command line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "train",
            "evaluate",
            "summary"
        ]
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    args = parser.parse_args()

    if (
        args.command in ("train", "evaluate")
        and args.fold is None
    ):
        parser.error(
            "--fold is required for train/evaluate"
        )

    if args.command == "train":

        train_fold(
            args.fold,
            args.epochs,
            args.lr
        )

    elif args.command == "evaluate":

        evaluate_fold(
            args.fold
        )

    else:

        summarize()


if __name__ == "__main__":
    main()
