#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scipy.ndimage import (
    zoom,
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
# Configuration
# ============================================================

CASE_ID = "Anev_04"
FOLD = 1
MODEL_NAME = "msau_unet3d_v2"

TARGET_SPACING = (1.5, 1.5, 1.5)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_FILE = (
    ROOT /
    f"0_raw_data/Anev/{CASE_ID}/nifti/"
    f"{CASE_ID}_image.nii.gz"
)

# Locate corrected GT only for evaluation, never for inference
gt_candidates = sorted(
    (ROOT / "0_raw_data_corrected").rglob(
        f"{CASE_ID}_segmentation_corrected.nii.gz"
    )
)

if len(gt_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one corrected GT for {CASE_ID}, "
        f"found {len(gt_candidates)}: {gt_candidates}"
    )

GT_FILE = gt_candidates[0]

CHECKPOINT = (
    ROOT /
    "5_checkpoints/folds_corrected_v5" /
    f"fold{FOLD}_{MODEL_NAME}_best.pth"
)

OUTDIR = (
    ROOT /
    "9_results/full_cta_no_gt_v1" /
    CASE_ID
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(pred, gt, spacing):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    union = np.logical_or(pred, gt).sum()

    dice = (
        2 * intersection / (pred_sum + gt_sum)
        if pred_sum + gt_sum > 0
        else 1.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    precision = (
        intersection / pred_sum
        if pred_sum > 0
        else 0.0
    )

    recall = (
        intersection / gt_sum
        if gt_sum > 0
        else 0.0
    )

    if pred_sum == 0 or gt_sum == 0:
        hd95 = float("nan")
        assd = float("nan")
    else:

        structure = np.ones((3, 3, 3), dtype=bool)

        pred_surface = pred & ~binary_erosion(
            pred, structure=structure
        )

        gt_surface = gt & ~binary_erosion(
            gt, structure=structure
        )

        dt_gt = distance_transform_edt(
            ~gt_surface,
            sampling=spacing
        )

        dt_pred = distance_transform_edt(
            ~pred_surface,
            sampling=spacing
        )

        d_pred_gt = dt_gt[pred_surface]
        d_gt_pred = dt_pred[gt_surface]

        distances = np.concatenate([
            d_pred_gt,
            d_gt_pred
        ])

        hd95 = float(
            np.percentile(distances, 95)
        )

        assd = float(
            distances.mean()
        )

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "HD95_mm": hd95,
        "ASSD_mm": assd,
        "Pred_voxels": int(pred_sum),
        "GT_voxels": int(gt_sum),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("FULL CTA — NO-GT ROI SINGLE CASE EVALUATION")
    print("=" * 100)

    print("Case       :", CASE_ID)
    print("Fold       :", FOLD)
    print("Model      :", MODEL_NAME)
    print("Device     :", DEVICE)
    print("Checkpoint :", CHECKPOINT)

    if not IMAGE_FILE.exists():
        raise FileNotFoundError(IMAGE_FILE)

    if not GT_FILE.exists():
        raise FileNotFoundError(GT_FILE)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    # --------------------------------------------------------
    # Load original CTA
    # --------------------------------------------------------

    nii = nib.load(str(IMAGE_FILE))

    image = nii.get_fdata(dtype=np.float32)

    original_shape = image.shape
    original_spacing = nii.header.get_zooms()[:3]

    factors = tuple(
        original_spacing[i] / TARGET_SPACING[i]
        for i in range(3)
    )

    print("\nOriginal shape   :", original_shape)
    print("Original spacing :", original_spacing)

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    image_rs = zoom(
        image,
        factors,
        order=1
    ).astype(np.float32)

    print("Resampled shape  :", image_rs.shape)

    # --------------------------------------------------------
    # HU clipping and full-volume normalization
    # --------------------------------------------------------

    image_rs = np.clip(
        image_rs,
        -200.0,
        400.0
    )

    mean = float(image_rs.mean())
    std = float(image_rs.std())

    image_rs = (
        (image_rs - mean) / (std + 1e-8)
    ).astype(np.float32)

    print("Normalization mean :", mean)
    print("Normalization std  :", std)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = train_mod.build_model(
        MODEL_NAME
    )

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)

    model = model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Full CTA inference — no GT input
    # --------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    prediction = train_mod.sliding_window_predict(
        model,
        image_rs,
        DEVICE
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start

    print("\nInference time (s):", round(inference_time, 2))

    # --------------------------------------------------------
    # Load GT only after inference
    # --------------------------------------------------------

    gt_nii = nib.load(str(GT_FILE))

    gt = (
        gt_nii.get_fdata() > 0
    ).astype(np.uint8)

    if gt.shape != original_shape:
        raise RuntimeError(
            f"GT shape mismatch: {gt.shape} vs {original_shape}"
        )

    if not np.allclose(
        gt_nii.affine,
        nii.affine,
        atol=1e-4
    ):
        raise RuntimeError(
            "GT affine mismatch"
        )

    gt_rs = zoom(
        gt,
        factors,
        order=0
    ) > 0

    if prediction.shape != gt_rs.shape:
        raise RuntimeError(
            f"Shape mismatch: "
            f"{prediction.shape} vs {gt_rs.shape}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    metrics = calculate_metrics(
        prediction,
        gt_rs,
        TARGET_SPACING
    )

    print("\n" + "=" * 100)
    print("FULL CTA — NO-GT EVALUATION RESULTS")
    print("=" * 100)

    for key, value in metrics.items():
        print(f"{key:18s}: {value}")

    # --------------------------------------------------------
    # Save prediction as NIfTI in resampled space
    # --------------------------------------------------------

    scale = np.diag([
        1 / factors[0],
        1 / factors[1],
        1 / factors[2],
        1
    ])

    affine_rs = nii.affine @ scale

    pred_nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        affine_rs
    )

    pred_path = (
        OUTDIR /
        f"{CASE_ID}_full_cta_prediction.nii.gz"
    )

    nib.save(
        pred_nii,
        str(pred_path)
    )

    import pandas as pd

    row = {
        "case_id": CASE_ID,
        "fold": FOLD,
        "model": MODEL_NAME,
        "normalization": "full_volume_zscore",
        "gt_guided_roi": False,
        "inference_time_s": inference_time,
        **metrics
    }

    pd.DataFrame([row]).to_csv(
        OUTDIR / "single_case_metrics.csv",
        index=False
    )

    print("\nPrediction saved:", pred_path)
    print("Metrics saved   :", OUTDIR / "single_case_metrics.csv")

    print("=" * 100)


if __name__ == "__main__":
    main()
