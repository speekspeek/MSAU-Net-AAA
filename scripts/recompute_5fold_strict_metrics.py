#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute case-level segmentation metrics for 5-fold experiments.

Metrics:
    Dice
    IoU
    Precision
    Recall
    HD95 (strict surface-based)
    ASSD (strict surface-based)

Important:
    - Metrics are calculated case-by-case.
    - HD95 / ASSD are calculated from binary-mask surfaces.
    - Physical spacing is used when available.
    - The script also checks suspicious duplicate predictions.

Expected fold CSV:
    2_dataset/folds/fold1_test.csv
    ...
    2_dataset/folds/fold5_test.csv

Prediction files may be .npy or .nii/.nii.gz.
You specify prediction root for each model.
"""

import os
import glob
import argparse
import hashlib

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    import nibabel as nib
except ImportError:
    nib = None


PROJECT_ROOT = os.path.expanduser("~/AAA_CTA_Project")


def load_array(path):
    """Load NPY or NIfTI and return ndarray + spacing."""
    if path.endswith(".npy"):
        arr = np.load(path)
        return np.asarray(arr), None

    if path.endswith(".nii") or path.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError(
                "nibabel is required for NIfTI files. "
                "Install with: pip install nibabel"
            )

        img = nib.load(path)
        arr = np.asarray(img.get_fdata())
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        return arr, spacing

    raise ValueError(f"Unsupported file: {path}")


def binarize(arr):
    """
    Robustly convert prediction/GT to binary mask.
    Handles:
      - probability mask
      - binary mask
      - logits-like arrays
      - singleton channels
    """
    arr = np.asarray(arr)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")

    if arr.dtype == bool:
        return arr

    unique = np.unique(arr)

    if len(unique) <= 3 and np.all(np.isin(unique, [0, 1, 255])):
        return arr > 0

    # probability-like output
    if np.nanmin(arr) >= 0 and np.nanmax(arr) <= 1:
        return arr >= 0.5

    return arr > 0


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return 2.0 * inter / denom


def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return inter / union


def precision_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    if tp + fp == 0:
        return 1.0 if gt.sum() == 0 else 0.0

    return tp / (tp + fp)


def recall_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    if tp + fn == 0:
        return 1.0

    return tp / (tp + fn)


def extract_surface(mask):
    """
    One-voxel internal surface.
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0
    )

    return np.logical_and(mask, ~eroded)


def strict_surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    Surface-to-surface distances in physical units.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return np.array([0.0]), np.array([0.0])

    if pred.sum() == 0 or gt.sum() == 0:
        return np.array([np.inf]), np.array([np.inf])

    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)

    # distance to GT surface
    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    # distance to prediction surface
    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    return d_pred_to_gt, d_gt_to_pred


def hd95_assd(pred, gt, spacing):
    d1, d2 = strict_surface_distances(pred, gt, spacing)

    if np.isinf(d1).any() or np.isinf(d2).any():
        return np.inf, np.inf

    all_dist = np.concatenate([d1, d2])

    hd95 = float(np.percentile(all_dist, 95))
    assd = float(
        (d1.sum() + d2.sum()) /
        (len(d1) + len(d2))
    )

    return hd95, assd


def sha1_array(mask):
    """
    Hash binary prediction for duplicate-prediction audit.
    """
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    return hashlib.sha1(mask.tobytes()).hexdigest()


def infer_case_id(row):
    for col in [
        "case_id",
        "case",
        "Case",
        "ID",
        "id"
    ]:
        if col in row.index:
            return str(row[col])

    return str(row.iloc[0])


def infer_group(row, case_id):
    for col in ["group", "Group", "label", "class"]:
        if col in row.index:
            value = str(row[col])

            if value.lower() in ["aaa", "normal"]:
                return value

    if case_id.startswith("Anev"):
        return "AAA"

    if case_id.startswith("Nat"):
        return "Normal"

    return "Unknown"


def find_gt(case_id):
    """
    Prefer original NIfTI segmentation to avoid ambiguity.
    """
    candidates = [
        os.path.join(
            PROJECT_ROOT,
            "3_preprocessed", "Anev", "labels",
            f"{case_id}_crop_lab.npy"
        ),
        os.path.join(
            PROJECT_ROOT,
            "0_raw_data", "Anev",
            case_id, "segmentation",
            f"{case_id}_segmentation.nii.gz"
        ),
        os.path.join(
            PROJECT_ROOT,
            "0_raw_data", "Nat",
            case_id, "segmentation",
            f"{case_id}_segmentation.nii.gz"
        ),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"GT not found for {case_id}"
    )


def find_prediction(pred_root, case_id):
    """
    Search for prediction file using several common naming patterns.
    """

    exact_names = [
        f"{case_id}.npy",
        f"{case_id}_pred.npy",
        f"{case_id}_prediction.npy",
        f"{case_id}_seg.npy",

        f"{case_id}.nii.gz",
        f"{case_id}_pred.nii.gz",
        f"{case_id}_prediction.nii.gz",
        f"{case_id}_seg.nii.gz",

        f"{case_id}.nii",
        f"{case_id}_pred.nii",
    ]

    for name in exact_names:
        matches = glob.glob(
            os.path.join(pred_root, "**", name),
            recursive=True
        )

        if matches:
            return sorted(matches)[0]

    # fallback
    matches = glob.glob(
        os.path.join(pred_root, "**", f"*{case_id}*"),
        recursive=True
    )

    matches = [
        p for p in matches
        if os.path.isfile(p)
        and (
            p.endswith(".npy")
            or p.endswith(".nii")
            or p.endswith(".nii.gz")
        )
    ]

    if not matches:
        raise FileNotFoundError(
            f"No prediction found for {case_id} under {pred_root}"
        )

    return sorted(matches)[0]


def match_shapes(pred, gt):
    """
    Only handle safe axis-order mismatch.
    No interpolation is performed here.
    """
    if pred.shape == gt.shape:
        return pred, gt

    # common NIfTI/NPY transpose possibility
    permutations = [
        (0, 1, 2),
        (2, 1, 0),
        (1, 0, 2),
        (0, 2, 1),
        (1, 2, 0),
        (2, 0, 1)
    ]

    for perm in permutations:
        p = np.transpose(pred, perm)
        if p.shape == gt.shape:
            print(
                f"  [shape] prediction transposed "
                f"{pred.shape} -> {p.shape}, perm={perm}"
            )
            return p, gt

    raise ValueError(
        f"Shape mismatch: prediction={pred.shape}, GT={gt.shape}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Model display name"
    )

    parser.add_argument(
        "--pred_root",
        required=True,
        help="Root folder containing 5-fold predictions"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV"
    )

    args = parser.parse_args()

    rows = []
    hashes = {}

    for fold in range(1, 6):

        fold_csv = os.path.join(
            PROJECT_ROOT,
            "2_dataset",
            "folds_dedup15",
            f"fold{fold}_test.csv"
        )

        if not os.path.isfile(fold_csv):
            raise FileNotFoundError(fold_csv)

        df = pd.read_csv(fold_csv)

        print("\n" + "=" * 80)
        print(f"Fold {fold}")
        print("=" * 80)

        for _, row in df.iterrows():

            case_id = infer_case_id(row)
            group = infer_group(row, case_id)

            gt_path = find_gt(case_id)
            pred_path = find_prediction(
                args.pred_root,
                case_id
            )

            print(f"\n[{case_id}]")
            print(f"GT   : {gt_path}")
            print(f"PRED : {pred_path}")

            gt, gt_spacing = load_array(gt_path)
            pred, pred_spacing = load_array(pred_path)

            gt = binarize(gt)
            pred = binarize(pred)

            pred, gt = match_shapes(pred, gt)

            # Prefer physical spacing from GT NIfTI.
            if gt_spacing is not None:
                spacing = gt_spacing
            elif pred_spacing is not None:
                spacing = pred_spacing
            else:
                # If cropped NPY has no spacing, infer from original GT.
                original_gt = None

                for raw_group in ["Anev", "Nat"]:
                    p = os.path.join(
                        PROJECT_ROOT,
                        "0_raw_data",
                        raw_group,
                        case_id,
                        "segmentation",
                        f"{case_id}_segmentation.nii.gz"
                    )

                    if os.path.isfile(p):
                        original_gt = p
                        break

                # Predictions and cropped GT are both in the
                # resampled preprocessing space: 1.5-mm isotropic.
                spacing = (1.5, 1.5, 1.5)

            dice = dice_score(pred, gt)
            iou = iou_score(pred, gt)
            precision = precision_score(pred, gt)
            recall = recall_score(pred, gt)

            hd95, assd = hd95_assd(
                pred,
                gt,
                spacing
            )

            pred_hash = sha1_array(pred)

            if pred_hash in hashes:
                print(
                    "\n*** WARNING ***"
                )
                print(
                    f"Prediction for {case_id} is IDENTICAL "
                    f"to {hashes[pred_hash]}"
                )
            else:
                hashes[pred_hash] = case_id

            print(
                f"Dice={dice:.6f} "
                f"IoU={iou:.6f} "
                f"HD95={hd95:.6f} "
                f"ASSD={assd:.6f}"
            )

            rows.append({
                "model": args.model,
                "fold": fold,
                "case_id": case_id,
                "group": group,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "hd95_mm": hd95,
                "assd_mm": assd,
                "pred_voxels": int(pred.sum()),
                "gt_voxels": int(gt.sum()),
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "prediction_path": pred_path,
                "gt_path": gt_path,
                "prediction_sha1": pred_hash
            })

    result = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(args.out),
        exist_ok=True
    )

    result.to_csv(
        args.out,
        index=False
    )

    print("\n\n===== CASE-LEVEL RESULTS =====")
    print(result[
        [
            "fold",
            "case_id",
            "dice",
            "iou",
            "hd95_mm",
            "assd_mm"
        ]
    ].to_string(index=False))

    summary = result[
        [
            "dice",
            "iou",
            "precision",
            "recall",
            "hd95_mm",
            "assd_mm"
        ]
    ].agg(["mean", "std"])

    summary_path = args.out.replace(
        ".csv",
        "_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n===== SUMMARY =====")
    print(summary)

    print("\nSaved:")
    print(args.out)
    print(summary_path)

    # duplicate audit
    duplicate = result[
        result.duplicated(
            "prediction_sha1",
            keep=False
        )
    ].sort_values("prediction_sha1")

    dup_path = args.out.replace(
        ".csv",
        "_duplicate_prediction_audit.csv"
    )

    duplicate.to_csv(
        dup_path,
        index=False
    )

    print("\n===== DUPLICATE PREDICTION AUDIT =====")

    if len(duplicate) == 0:
        print("No identical prediction masks detected.")
    else:
        print(
            duplicate[
                [
                    "fold",
                    "case_id",
                    "prediction_path",
                    "prediction_sha1"
                ]
            ].to_string(index=False)
        )

    print("\nAudit saved:")
    print(dup_path)


if __name__ == "__main__":
    main()
