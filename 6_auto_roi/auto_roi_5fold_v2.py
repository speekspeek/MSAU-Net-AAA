#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AAA automatic ROI localization V2.

Uses:
  - Existing V1 locator checkpoints
  - Existing frozen Full MSAU-Net checkpoints
  - Internal validation cases for ROI calibration
  - Outer-test GT only for final evaluation

Commands:
  python 6_auto_roi/auto_roi_5fold_v2.py calibrate --fold 1
  python 6_auto_roi/auto_roi_5fold_v2.py evaluate --fold 1
  python 6_auto_roi/auto_roi_5fold_v2.py summary
"""

import argparse
import ast
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import (
    label as cc_label,
    find_objects,
)

ROOT = Path.home() / "AAA_CTA_Project"

V1_PATH = (
    ROOT / "6_auto_roi/auto_roi_5fold.py"
)

OUT = (
    ROOT / "9_results/auto_roi_5fold_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_SPACING = (1.5, 1.5, 1.5)

# Prespecified candidate grid.
# Final choice is made using internal validation only.

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

MARGINS = [24, 40, 56]

COMPONENT_POLICIES = [
    "largest",
    "top3",
    "all",
]

# Constraint for validation ROI volume.
MAX_ROI_FRACTION = 0.65

# Coverage target used in validation selection.
TARGET_COVERAGE = 0.98


# ============================================================
# Load existing V1 module without executing its CLI
# ============================================================

spec = importlib.util.spec_from_file_location(
    "aaa_auto_roi_v1",
    str(V1_PATH)
)

v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def device_name():

    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def get_locator(fold, device):

    checkpoint = (
        v1.MODEL_DIR /
        f"fold{fold}_locator_best.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Train V1 locator for fold {fold} first: "
            f"{checkpoint}"
        )

    model = v1.LocatorUNet().to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


def locator_probability(image, model, device):

    x = torch.from_numpy(
        v1.coarse_image(image)[None, None].copy()
    ).to(device)

    with torch.no_grad():

        probability = torch.sigmoid(
            model(x)
        )[0, 0].cpu().numpy()

    return probability


# ============================================================
# Automatic ROI generation
# No GT is used in this function.
# ============================================================

def generate_roi(
    probability,
    full_shape,
    threshold,
    margin,
    policy
):

    coarse_mask = (
        probability >= threshold
    )

    structure = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    components, num = cc_label(
        coarse_mask,
        structure=structure
    )

    shape = np.asarray(
        full_shape,
        dtype=int
    )

    # Empty locator: fall back to full image.

    if num == 0:

        return (
            np.zeros(3, dtype=int),
            shape.copy(),
            "empty_full_volume"
        )

    sizes = np.bincount(
        components.ravel()
    )

    sizes[0] = 0

    ranked = np.argsort(
        sizes[1:]
    )[::-1] + 1

    if policy == "largest":

        selected = ranked[:1]

    elif policy == "top3":

        selected = ranked[:3]

    elif policy == "all":

        selected = ranked

    else:

        raise ValueError(policy)

    # Use component bounding boxes rather than
    # allocating one full-size boolean mask per component.

    objects = find_objects(components)

    lows = []
    highs = []

    for component_id in selected:

        obj = objects[int(component_id) - 1]

        if obj is None:
            continue

        lows.append([
            s.start for s in obj
        ])

        highs.append([
            s.stop for s in obj
        ])

    if not lows:

        return (
            np.zeros(3, dtype=int),
            shape.copy(),
            "empty_components_full_volume"
        )

    low_coarse = np.min(
        np.asarray(lows),
        axis=0
    )

    high_coarse = np.max(
        np.asarray(highs),
        axis=0
    )

    coarse_shape = np.asarray(
        probability.shape,
        dtype=float
    )

    low = np.floor(
        low_coarse * shape / coarse_shape
    ).astype(int)

    high = np.ceil(
        high_coarse * shape / coarse_shape
    ).astype(int)

    low = np.maximum(
        low - margin,
        0
    )

    high = np.minimum(
        high + margin,
        shape
    )

    if np.any(high <= low):
        raise RuntimeError(
            "Invalid automatic ROI"
        )

    return low, high, policy


def roi_fraction(low, high, shape):

    return float(
        np.prod(
            (high - low)
            /
            np.asarray(shape, dtype=float)
        )
    )


def gt_coverage(gt, low, high):

    total = int(gt.sum())

    if total == 0:
        return float("nan")

    inside = int(
        gt[
            low[0]:high[0],
            low[1]:high[1],
            low[2]:high[2]
        ].sum()
    )

    return inside / total


# ============================================================
# Internal validation calibration
# ============================================================

def calibrate_fold(fold):

    cases = v1.verify_cases()

    train_cases, val_cases = (
        v1.split_train_val(
            fold,
            cases
        )
    )

    print("=" * 100)
    print("V2 INTERNAL ROI CALIBRATION")
    print("=" * 100)

    print("Fold:", fold)
    print("Training cases:", train_cases)
    print("Validation cases:", val_cases)
    print("Outer test cases:", v1.FOLDS[fold])

    device = device_name()

    locator = get_locator(
        fold,
        device
    )

    # Only internal validation cases are loaded.
    # Outer-test cases are not accessed.

    val_data = []

    for case in val_cases:

        print(
            "Preparing validation:",
            case,
            flush=True
        )

        image, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        probability = locator_probability(
            image,
            locator,
            device
        )

        val_data.append({
            "case": case,
            "gt": gt,
            "probability": probability,
            "shape": image.shape
        })

        del image

    candidates = []

    for threshold, margin, policy in itertools.product(
        THRESHOLDS,
        MARGINS,
        COMPONENT_POLICIES
    ):

        coverages = []
        fractions = []

        for item in val_data:

            low, high, _ = generate_roi(
                item["probability"],
                item["shape"],
                threshold,
                margin,
                policy
            )

            coverage = gt_coverage(
                item["gt"],
                low,
                high
            )

            fraction = roi_fraction(
                low,
                high,
                item["shape"]
            )

            coverages.append(coverage)
            fractions.append(fraction)

        candidates.append({
            "threshold": threshold,
            "margin": margin,
            "policy": policy,
            "mean_coverage": float(
                np.mean(coverages)
            ),
            "min_coverage": float(
                np.min(coverages)
            ),
            "mean_roi_fraction": float(
                np.mean(fractions)
            ),
            "max_roi_fraction": float(
                np.max(fractions)
            )
        })

    result = pd.DataFrame(candidates)

    fold_out = OUT / f"fold{fold}"

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        fold_out / "calibration_grid.csv",
        index=False
    )

    # Prespecified selection procedure:
    #
    # 1. Require ROI volume <= MAX_ROI_FRACTION
    #    on every validation case, if feasible.
    # 2. Prefer minimum coverage >= TARGET_COVERAGE.
    # 3. Otherwise maximize minimum coverage.
    # 4. Break ties by mean coverage and ROI volume.
    #
    # No outer-test metrics are used.

    feasible = result[
        result["max_roi_fraction"]
        <= MAX_ROI_FRACTION
    ].copy()

    if feasible.empty:

        feasible = result.copy()
        constraint_status = (
            "no_candidate_satisfies_volume_constraint"
        )

    else:

        constraint_status = "volume_constraint_satisfied"

    target_met = feasible[
        feasible["min_coverage"]
        >= TARGET_COVERAGE
    ].copy()

    if not target_met.empty:

        chosen = target_met.sort_values(
            [
                "mean_roi_fraction",
                "min_coverage",
                "mean_coverage"
            ],
            ascending=[
                True,
                False,
                False
            ]
        ).iloc[0]

        selection_status = "coverage_target_met"

    else:

        chosen = feasible.sort_values(
            [
                "min_coverage",
                "mean_coverage",
                "mean_roi_fraction"
            ],
            ascending=[
                False,
                False,
                True
            ]
        ).iloc[0]

        selection_status = "best_available_coverage"

    config = {
        "fold": int(fold),
        "threshold": float(chosen["threshold"]),
        "margin": int(chosen["margin"]),
        "policy": str(chosen["policy"]),
        "validation_mean_coverage": float(
            chosen["mean_coverage"]
        ),
        "validation_min_coverage": float(
            chosen["min_coverage"]
        ),
        "validation_mean_roi_fraction": float(
            chosen["mean_roi_fraction"]
        ),
        "validation_max_roi_fraction": float(
            chosen["max_roi_fraction"]
        ),
        "constraint_status": constraint_status,
        "selection_status": selection_status,
        "validation_cases": val_cases,
        "training_cases": train_cases
    }

    config_file = (
        fold_out / "selected_roi_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSelected configuration:")
    print(json.dumps(config, indent=2))

    print("\nSaved:", config_file)
    print("=" * 100)


# ============================================================
# Frozen Full MSAU-Net
# ============================================================

def load_fine_model(fold, device):

    sys.path.insert(
        0,
        str(ROOT / "4_train")
    )

    import train_corrected_v3_fold1_sanity as train_mod

    checkpoint = (
        ROOT /
        "5_checkpoints/folds_corrected_v5" /
        f"fold{fold}_msau_unet3d_v2_best.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = train_mod.build_model(
        "msau_unet3d_v2"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    return model, train_mod


# ============================================================
# Outer-test evaluation
# ============================================================

def evaluate_fold(fold):

    v1.verify_cases()

    fold_out = OUT / f"fold{fold}"

    config_file = (
        fold_out / "selected_roi_config.json"
    )

    if not config_file.exists():
        raise FileNotFoundError(
            "Run calibrate first: "
            + str(config_file)
        )

    with open(
        config_file,
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    if int(config["fold"]) != fold:
        raise RuntimeError(
            "Fold configuration mismatch"
        )

    device = device_name()

    locator = get_locator(
        fold,
        device
    )

    fine, train_mod = load_fine_model(
        fold,
        device
    )

    print("=" * 100)
    print("V2 AUTOMATIC ROI — OUTER TEST")
    print("=" * 100)

    print("Fold:", fold)
    print("Config:", config)
    print("Device:", device)

    rows = []

    for case in v1.FOLDS[fold]:

        print(
            "\nEvaluating:",
            case,
            flush=True
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # GT is not loaded during ROI generation
        # or fine segmentation.

        image, _, affine, factors = (
            v1.load_volume(
                case,
                need_gt=False
            )
        )

        probability = locator_probability(
            image,
            locator,
            device
        )

        low, high, status = generate_roi(
            probability,
            image.shape,
            config["threshold"],
            config["margin"],
            config["policy"]
        )

        crop = image[
            low[0]:high[0],
            low[1]:high[1],
            low[2]:high[2]
        ].copy()

        mean = float(crop.mean())
        std = float(crop.std())

        crop = (
            (crop - mean)
            /
            (std + 1e-8)
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

        elapsed = time.perf_counter() - t0

        # Save prediction before GT-based evaluation.

        rescale_affine = affine @ np.diag([
            1 / factors[0],
            1 / factors[1],
            1 / factors[2],
            1
        ])

        prediction_path = (
            fold_out /
            f"{case}_prediction.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(
                pred,
                rescale_affine
            ),
            str(prediction_path)
        )

        # GT is used only for evaluation.

        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        coverage = gt_coverage(
            gt,
            low,
            high
        )

        result = v1.metrics(
            pred,
            gt
        )

        row = {
            "case_id": case,
            "fold": fold,
            "threshold": config["threshold"],
            "margin": config["margin"],
            "policy": config["policy"],
            "roi_status": status,
            "roi_start": str(low.tolist()),
            "roi_stop_exclusive": str(high.tolist()),
            "roi_fraction": roi_fraction(
                low,
                high,
                image.shape
            ),
            "roi_gt_coverage": coverage,
            "inference_seconds": elapsed,
            **result
        }

        rows.append(row)

        pd.DataFrame(rows).to_csv(
            fold_out / "metrics.csv",
            index=False
        )

        print(
            f"{case} "
            f"Dice={result['Dice']:.4f} "
            f"Recall={result['Recall']:.4f} "
            f"ROI_Coverage={coverage:.4f} "
            f"ROI_Fraction={row['roi_fraction']:.4f} "
            f"HD95={result['HD95_mm']:.2f}",
            flush=True
        )

        del image, gt, crop, pred_crop, pred, probability

    print("\nSaved:", fold_out / "metrics.csv")


# ============================================================
# 15-case summary
# ============================================================

def summary():

    v1.verify_cases()

    files = [
        OUT / f"fold{fold}" / "metrics.csv"
        for fold in range(1, 6)
    ]

    missing = [
        str(p)
        for p in files
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing fold metrics: "
            + str(missing)
        )

    df = pd.concat(
        [pd.read_csv(p) for p in files],
        ignore_index=True
    )

    if (
        len(df) != 15
        or df["case_id"].nunique() != 15
    ):
        raise RuntimeError(
            "Expected exactly 15 unique OOF cases"
        )

    for fold in range(1, 6):

        actual = set(
            df.loc[
                df["fold"] == fold,
                "case_id"
            ].astype(str)
        )

        if actual != set(v1.FOLDS[fold]):
            raise RuntimeError(
                f"Fold {fold}: test cases mismatch"
            )

    df.to_csv(
        OUT / "all_15_oof_metrics.csv",
        index=False
    )

    cols = [
        "roi_gt_coverage",
        "roi_fraction",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "HD95_mm",
        "ASSD_mm"
    ]

    result = (
        df[cols]
        .agg(["mean", "std", "min", "max"])
        .T
    )

    result["valid_n"] = df[cols].count()

    result.to_csv(
        OUT / "summary.csv"
    )

    print("=" * 100)
    print("V2 — 15-CASE OOF SUMMARY")
    print("=" * 100)

    print(result.to_string())

    print("\nSaved:", OUT / "summary.csv")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "calibrate",
            "evaluate",
            "summary"
        ]
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6)
    )

    args = parser.parse_args()

    if args.command in (
        "calibrate",
        "evaluate"
    ) and args.fold is None:

        parser.error(
            "--fold is required"
        )

    if args.command == "calibrate":

        calibrate_fold(args.fold)

    elif args.command == "evaluate":

        evaluate_fold(args.fold)

    else:

        summary()


if __name__ == "__main__":
    main()
