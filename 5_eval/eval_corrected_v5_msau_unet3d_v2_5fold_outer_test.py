#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt
)


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

sys.path.insert(
    0,
    str(ROOT / "4_train")
)

import train_corrected_v3_fold1_sanity as train_mod


# ============================================================
# Fixed experiment configuration
# ============================================================

MODEL_NAME = "msau_unet3d_v2"

FOLDS = [1, 2, 3, 4, 5]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SPACING = (
    1.5,
    1.5,
    1.5
)

FOLD_ROOT = (
    ROOT /
    "2_dataset/folds_dedup15_v3_corrected"
)

CHECKPOINT_ROOT = (
    ROOT /
    "5_checkpoints/folds_corrected_v5"
)

OUT_DIR = (
    ROOT /
    "9_results/corrected_v5_oof/msau_unet3d_v2"
)

PRED_DIR = (
    OUT_DIR /
    "predictions"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Metrics
# ============================================================

def dice_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    denom = (
        pred.sum()
        +
        gt.sum()
    )

    if denom == 0:
        return 1.0

    return float(
        2.0 * inter / denom
    )


def iou_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    if union == 0:
        return 1.0

    return float(
        inter / union
    )


def precision_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(
        pred,
        gt
    ).sum()

    fp = np.logical_and(
        pred,
        ~gt
    ).sum()

    if tp + fp == 0:
        return (
            1.0
            if gt.sum() == 0
            else 0.0
        )

    return float(
        tp / (tp + fp)
    )


def recall_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(
        pred,
        gt
    ).sum()

    fn = np.logical_and(
        ~pred,
        gt
    ).sum()

    if tp + fn == 0:
        return 1.0

    return float(
        tp / (tp + fn)
    )


# ============================================================
# Strict surface metrics
# ============================================================

def extract_surface(mask):

    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    eroded = binary_erosion(
        mask,
        structure=np.ones(
            (3, 3, 3),
            dtype=bool
        ),
        border_value=0
    )

    return np.logical_and(
        mask,
        ~eroded
    )


def hd95_assd(
    pred,
    gt,
    spacing=SPACING
):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if (
        pred.sum() == 0
        and
        gt.sum() == 0
    ):
        return 0.0, 0.0

    if (
        pred.sum() == 0
        or
        gt.sum() == 0
    ):
        return np.inf, np.inf

    pred_surface = extract_surface(
        pred
    )

    gt_surface = extract_surface(
        gt
    )

    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    d_pred_gt = dt_gt[
        pred_surface
    ]

    d_gt_pred = dt_pred[
        gt_surface
    ]

    all_dist = np.concatenate(
        [
            d_pred_gt,
            d_gt_pred
        ]
    )

    hd95 = float(
        np.percentile(
            all_dist,
            95
        )
    )

    assd = float(
        (
            d_pred_gt.sum()
            +
            d_gt_pred.sum()
        )
        /
        (
            len(d_pred_gt)
            +
            len(d_gt_pred)
        )
    )

    return hd95, assd


# ============================================================
# Load model
# ============================================================

print("=" * 100)
print("CORRECTED V5 — FULL MSAU-NET 5-FOLD OUTER-TEST OOF EVALUATION")
print("=" * 100)

print("Model      :", MODEL_NAME)
print("Folds      :", FOLDS)
print("Device     :", DEVICE)
print("Patch size :", train_mod.SW_PATCH_SIZE)
print("Stride     :", train_mod.SW_STRIDE)
print("Threshold  : 0.5")
print("Spacing    :", SPACING)
print("=" * 100)


all_rows = []


# ============================================================
# Evaluate five outer-test folds
# ============================================================

for fold in FOLDS:

    print()
    print("=" * 100)
    print(f"FOLD {fold} OUTER-TEST")
    print("=" * 100)

    test_csv = (
        FOLD_ROOT /
        f"fold{fold}_test.csv"
    )

    checkpoint = (
        CHECKPOINT_ROOT /
        f"fold{fold}_{MODEL_NAME}_best.pth"
    )

    if not test_csv.exists():
        raise FileNotFoundError(
            test_csv
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    test_df = pd.read_csv(
        test_csv
    )

    if len(test_df) != 3:
        raise RuntimeError(
            f"Fold {fold}: expected 3 outer-test cases, "
            f"got {len(test_df)}"
        )

    print("Checkpoint :", checkpoint)
    print(
        "Test cases :",
        ", ".join(
            test_df["case_id"].astype(str).tolist()
        )
    )

    # --------------------------------------------------------
    # Fresh model for each fold
    # --------------------------------------------------------

    model = train_mod.build_model(
        MODEL_NAME
    )

    state = torch.load(
        checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    model = model.to(
        DEVICE
    )

    model.eval()


    # --------------------------------------------------------
    # Outer-test cases
    # --------------------------------------------------------

    with torch.no_grad():

        for _, row in test_df.iterrows():

            case_id = str(
                row["case_id"]
            )

            image = np.load(
                ROOT /
                row["image_npy"]
            ).astype(
                np.float32
            )

            gt = np.load(
                ROOT /
                row["label_npy"]
            )

            gt = (
                gt > 0
            )

            prediction = (
                train_mod.sliding_window_predict(
                    model,
                    image,
                    DEVICE
                )
            )

            prediction = (
                prediction > 0
            )

            if prediction.shape != gt.shape:
                raise RuntimeError(
                    f"{case_id}: "
                    f"prediction={prediction.shape}, "
                    f"GT={gt.shape}"
                )

            # ------------------------------------------------
            # Save prediction
            # ------------------------------------------------

            fold_pred_dir = (
                PRED_DIR /
                f"fold{fold}"
            )

            fold_pred_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            pred_path = (
                fold_pred_dir /
                f"{case_id}_pred.npy"
            )

            np.save(
                pred_path,
                prediction.astype(
                    np.uint8
                )
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            dice = dice_score(
                prediction,
                gt
            )

            iou = iou_score(
                prediction,
                gt
            )

            precision = precision_score(
                prediction,
                gt
            )

            recall = recall_score(
                prediction,
                gt
            )

            hd95, assd = hd95_assd(
                prediction,
                gt,
                SPACING
            )

            pred_voxels = int(
                prediction.sum()
            )

            gt_voxels = int(
                gt.sum()
            )

            ratio = (
                pred_voxels / gt_voxels
                if gt_voxels > 0
                else np.nan
            )

            result = {
                "fold":
                    fold,

                "case_id":
                    case_id,

                "class":
                    row.get(
                        "class",
                        ""
                    ),

                "dice":
                    dice,

                "iou":
                    iou,

                "precision":
                    precision,

                "recall":
                    recall,

                "hd95_mm":
                    hd95,

                "assd_mm":
                    assd,

                "pred_voxels":
                    pred_voxels,

                "gt_voxels":
                    gt_voxels,

                "pred_gt_ratio":
                    ratio
            }

            all_rows.append(
                result
            )

            print(
                f"{case_id:10s} "
                f"Dice={dice:.6f} "
                f"IoU={iou:.6f} "
                f"P={precision:.6f} "
                f"R={recall:.6f} "
                f"HD95={hd95:.3f} "
                f"ASSD={assd:.3f} "
                f"Pred/GT={ratio:.3f}"
            )


# ============================================================
# Build final 15-case OOF table
# ============================================================

results = pd.DataFrame(
    all_rows
)

if len(results) != 15:
    raise RuntimeError(
        f"Expected 15 OOF cases, got {len(results)}"
    )

if results["case_id"].nunique() != 15:
    raise RuntimeError(
        "OOF case IDs are not unique. "
        "Possible fold overlap detected."
    )


metric_cols = [
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95_mm",
    "assd_mm"
]


# ============================================================
# Save per-case OOF
# ============================================================

case_csv = (
    OUT_DIR /
    "msau_unet3d_v2_5fold_oof_per_case.csv"
)

results.to_csv(
    case_csv,
    index=False
)


# ============================================================
# Per-fold summary
# ============================================================

per_fold = (
    results
    .groupby("fold")[metric_cols]
    .agg(["mean", "std"])
)

per_fold_csv = (
    OUT_DIR /
    "msau_unet3d_v2_5fold_oof_per_fold.csv"
)

per_fold.to_csv(
    per_fold_csv
)


# ============================================================
# Overall 15-case OOF summary
# ============================================================

summary = (
    results[metric_cols]
    .agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max"
        ]
    )
)

summary_csv = (
    OUT_DIR /
    "msau_unet3d_v2_5fold_oof_summary.csv"
)

summary.to_csv(
    summary_csv
)


# ============================================================
# AAA / Normal descriptive summary
# ============================================================

class_summary = (
    results
    .groupby("class")[metric_cols]
    .agg(["mean", "std"])
)

class_csv = (
    OUT_DIR /
    "msau_unet3d_v2_5fold_oof_by_class.csv"
)

class_summary.to_csv(
    class_csv
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("FINAL FULL MSAU-NET 15-CASE OOF RESULTS")
print("=" * 100)

print(
    results[
        [
            "fold",
            "case_id",
            "class",
            "dice",
            "iou",
            "precision",
            "recall",
            "hd95_mm",
            "assd_mm"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)

print()
print("=" * 100)
print("OVERALL OOF SUMMARY")
print("=" * 100)

print(
    summary
    .round(6)
    .to_string()
)

print()
print("=" * 100)
print("MEAN ± SD")
print("=" * 100)

for metric in metric_cols:

    mean = results[metric].mean()
    std = results[metric].std()

    print(
        f"{metric:12s}: "
        f"{mean:.6f} ± {std:.6f}"
    )


print()
print("Saved:")
print(case_csv)
print(per_fold_csv)
print(summary_csv)
print(class_csv)
print(PRED_DIR)

print("=" * 100)
print("FULL MSAU-NET 5-FOLD OOF EVALUATION FINISHED")
print("=" * 100)

