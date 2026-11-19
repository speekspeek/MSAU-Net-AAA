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

MODEL_NAME = "unet3d"
FOLD = 1

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

TEST_CSV = (
    ROOT /
    "2_dataset/folds_dedup15_v3_corrected/"
    "fold1_test.csv"
)

CHECKPOINT = (
    ROOT /
    "5_checkpoints/corrected_v3_sanity/"
    "fold1_unet3d_best.pth"
)

OUT_DIR = (
    ROOT /
    "9_results/corrected_v3_sanity/"
    "fold1_unet3d_outer_test"
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
print("CORRECTED V3 — FOLD 1 OUTER-TEST SANITY EVALUATION")
print("=" * 100)

print("Model      :", MODEL_NAME)
print("Fold       :", FOLD)
print("Device     :", DEVICE)
print("Checkpoint :", CHECKPOINT)
print("Patch size :", train_mod.SW_PATCH_SIZE)
print("Stride     :", train_mod.SW_STRIDE)
print("Threshold  : 0.5")
print("Spacing    :", SPACING)
print("=" * 100)


if not CHECKPOINT.exists():
    raise FileNotFoundError(
        CHECKPOINT
    )


model = train_mod.build_model(
    MODEL_NAME
)

state = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(
    state
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# Outer-test
# ============================================================

df = pd.read_csv(
    TEST_CSV
)

if len(df) != 3:
    raise RuntimeError(
        f"Expected 3 outer-test cases, got {len(df)}"
    )


rows = []


with torch.no_grad():

    for _, row in df.iterrows():

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

        if (
            prediction.shape
            !=
            gt.shape
        ):
            raise RuntimeError(
                f"{case_id}: "
                f"prediction={prediction.shape}, "
                f"GT={gt.shape}"
            )


        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        pred_path = (
            PRED_DIR /
            f"{case_id}_pred.npy"
        )

        np.save(
            pred_path,
            prediction.astype(
                np.uint8
            )
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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
            pred_voxels /
            gt_voxels
            if gt_voxels > 0
            else np.nan
        )

        rows.append({
            "fold":
                FOLD,

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
        })


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
# Save results
# ============================================================

results = pd.DataFrame(
    rows
)

case_csv = (
    OUT_DIR /
    "fold1_unet3d_outer_test_metrics.csv"
)

results.to_csv(
    case_csv,
    index=False
)


summary = (
    results[
        [
            "dice",
            "iou",
            "precision",
            "recall",
            "hd95_mm",
            "assd_mm"
        ]
    ]
    .agg(
        [
            "mean",
            "std"
        ]
    )
)

summary_csv = (
    OUT_DIR /
    "fold1_unet3d_outer_test_summary.csv"
)

summary.to_csv(
    summary_csv
)


print("\n" + "=" * 100)
print("OUTER-TEST SUMMARY")
print("=" * 100)

print(
    summary.round(
        6
    ).to_string()
)

print("\nSaved:")
print(case_csv)
print(summary_csv)
print(PRED_DIR)

print("=" * 100)
