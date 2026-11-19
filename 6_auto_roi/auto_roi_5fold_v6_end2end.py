#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V6 END-TO-END AAA CTA SEGMENTATION

Pipeline:
    Full CTA
      -> V3 final locator
      -> V6 multi-candidate segmented ROIs
      -> Frozen Full MSAU-Net
      -> Merge ROI predictions
      -> Full-volume segmentation
      -> GT-based evaluation

No GT used during inference.

Run:
    python -u 6_auto_roi/auto_roi_5fold_v6_end2end.py --fold 1

Outputs:
    9_results/auto_roi_5fold_v6_end2end/fold1/
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch


ROOT = Path.home() / "AAA_CTA_Project"

V1_PATH = ROOT / "6_auto_roi/auto_roi_5fold.py"

V6_PATH = ROOT / "6_auto_roi/auto_roi_5fold_v6.py"

V3_ROOT = ROOT / "9_results/auto_roi_5fold_v3"

V6_ROOT = ROOT / "9_results/auto_roi_5fold_v6"

OUT = ROOT / "9_results/auto_roi_5fold_v6_end2end"


# ============================================================
# Import existing modules
# ============================================================

def import_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


v1 = import_module(
    "aaa_auto_v1",
    V1_PATH
)

v6 = import_module(
    "aaa_auto_v6",
    V6_PATH
)


# ============================================================
# Load models
# ============================================================

def load_locator(fold, device):

    checkpoint = (
        V3_ROOT /
        f"fold{fold}" /
        "final_locator_12cases.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = v1.LocatorUNet()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    print("Locator:", checkpoint, flush=True)

    return model


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

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    print("Fine model:", checkpoint, flush=True)

    return model, train_mod


# ============================================================
# Coarse probability
# ============================================================

def locator_predict(model, image, device):

    coarse = v1.coarse_image(image)

    x = torch.from_numpy(
        coarse[None, None].copy()
    ).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(x)

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    return probability.astype(np.float32)


# ============================================================
# Load fixed V6 parameters
# ============================================================

def load_config(fold):

    path = (
        V6_ROOT /
        f"fold{fold}" /
        "selected_roi_config.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    if int(config["fold"]) != fold:
        raise RuntimeError(
            "ROI configuration fold mismatch"
        )

    if config["status"] != "FEASIBLE_CONFIGURATION_FOUND":
        raise RuntimeError(
            "V6 ROI configuration is not feasible"
        )

    required = [
        "axis",
        "slab",
        "threshold",
        "margin",
        "top_k"
    ]

    for key in required:
        if key not in config:
            raise RuntimeError(
                f"Missing ROI parameter: {key}"
            )

    return config


# ============================================================
# Validate boxes
# ============================================================

def validate_boxes(boxes, shape):

    shape = np.asarray(shape, dtype=int)

    for i, box in enumerate(boxes):

        low = np.asarray(
            box["low"],
            dtype=int
        )

        high = np.asarray(
            box["high"],
            dtype=int
        )

        if (
            low.shape != (3,)
            or high.shape != (3,)
            or np.any(low < 0)
            or np.any(high > shape)
            or np.any(high <= low)
        ):
            raise RuntimeError(
                f"Invalid ROI box {i}: {box}"
            )


# ============================================================
# Fine segmentation
# ============================================================

def segment_rois(
    image,
    boxes,
    fine_model,
    train_mod,
    device
):
    """
    Fine inference for each ROI.

    The existing V5 sliding-window function
    returns a binary segmentation.

    Therefore overlapping ROI predictions
    are merged by binary-vote averaging,
    not by unavailable soft probabilities.

    Voxels not covered by any ROI remain zero.
    """

    # uint16 is sufficient for the expected
    # number of overlapping boxes.

    vote_sum = np.zeros(
        image.shape,
        dtype=np.uint16
    )

    vote_count = np.zeros(
        image.shape,
        dtype=np.uint16
    )

    total_box_voxels = 0

    for i, box in enumerate(boxes, start=1):

        low = np.asarray(
            box["low"],
            dtype=int
        )

        high = np.asarray(
            box["high"],
            dtype=int
        )

        slices = tuple(
            slice(int(low[a]), int(high[a]))
            for a in range(3)
        )

        crop = image[slices].copy()

        mean = float(crop.mean())
        std = float(crop.std())

        crop = (
            (crop - mean) / (std + 1e-8)
        ).astype(np.float32)

        with torch.no_grad():

            pred_crop = (
                train_mod.sliding_window_predict(
                    fine_model,
                    crop,
                    device
                )
            )

        pred_crop = np.asarray(
            pred_crop
        )

        if pred_crop.shape != crop.shape:
            raise RuntimeError(
                f"ROI {i}: prediction shape mismatch "
                f"{pred_crop.shape} vs {crop.shape}"
            )

        vote_sum[slices] += (
            pred_crop > 0
        ).astype(np.uint16)

        vote_count[slices] += 1

        total_box_voxels += int(
            np.prod(high - low)
        )

        print(
            f"  ROI {i:02d}/{len(boxes)} "
            f"shape={tuple(crop.shape)} "
            f"positive={int((pred_crop > 0).sum())}",
            flush=True
        )

        del crop, pred_crop

    # Majority vote in overlapping regions.
    # A tie is resolved as positive.
    # This rule is fixed before evaluation.

    pred = (
        (vote_count > 0)
        &
        (vote_sum.astype(np.uint32) * 2
         >= vote_count.astype(np.uint32))
    ).astype(np.uint8)

    union_voxels = int(
        np.count_nonzero(vote_count)
    )

    if union_voxels > total_box_voxels:
        raise RuntimeError(
            "ROI union volume exceeds summed volume"
        )

    roi_fraction = (
        union_voxels / image.size
    )

    total_processing_fraction = (
        total_box_voxels / image.size
    )

    # ROI union is saved for later audit.
    union_mask = (
        vote_count > 0
    )

    del vote_sum, vote_count

    return (
        pred,
        union_mask,
        roi_fraction,
        total_processing_fraction
    )


# ============================================================
# GT audit
# ============================================================

def evaluate_prediction(
    pred,
    union_mask,
    gt
):

    if pred.shape != gt.shape:
        raise RuntimeError(
            "Prediction/GT shape mismatch"
        )

    total_gt = int(gt.sum())

    inside = int(
        (union_mask & gt).sum()
    )

    coverage = (
        inside / total_gt
        if total_gt > 0
        else float("nan")
    )

    result = v1.metrics(
        pred,
        gt
    )

    result["roi_gt_coverage"] = coverage
    result["gt_outside_roi"] = total_gt - inside

    return result


# ============================================================
# Save resampled NIfTI
# ============================================================

def save_prediction(
    prediction,
    affine,
    factors,
    output_path
):

    scale = np.diag([
        1.0 / factors[0],
        1.0 / factors[1],
        1.0 / factors[2],
        1.0
    ])

    resampled_affine = affine @ scale

    nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        resampled_affine
    )

    nii.header.set_zooms(
        v1.TARGET_SPACING
    )

    nib.save(
        nii,
        str(output_path)
    )


# ============================================================
# Run outer-test fold
# ============================================================

def evaluate_fold(fold):

    v1.verify_cases()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        print(
            "WARNING: CUDA unavailable; CPU inference "
            "may be extremely slow",
            flush=True
        )

    fold_out = OUT / f"fold{fold}"

    fold_out.mkdir(
        parents=True,
        exist_ok=True
    )

    config = load_config(fold)

    locator = load_locator(
        fold,
        device
    )

    fine_model, train_mod = load_fine_model(
        fold,
        device
    )

    print("=" * 100)
    print("V6 END-TO-END FULL CTA SEGMENTATION")
    print("=" * 100)

    print("Fold:", fold)
    print("Device:", device)
    print("Outer-test cases:", v1.FOLDS[fold])
    print("ROI config:", json.dumps(config))
    print("=" * 100)

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

        # ----------------------------------------------------
        # Step 1: Load image only
        # ----------------------------------------------------

        image, _, affine, factors = (
            v1.load_volume(
                case,
                need_gt=False
            )
        )

        print(
            "Full CTA shape:",
            image.shape,
            flush=True
        )

        # ----------------------------------------------------
        # Step 2: Coarse locator
        # ----------------------------------------------------

        probability = locator_predict(
            locator,
            image,
            device
        )

        # ----------------------------------------------------
        # Step 3: V6 ROI generation
        # ----------------------------------------------------

        candidates = v6.component_candidates(
            probability,
            float(config["threshold"]),
            int(config["axis"]),
            int(config["slab"])
        )

        boxes = v6.build_boxes(
            candidates,
            probability.shape,
            image.shape,
            config
        )

        validate_boxes(
            boxes,
            image.shape
        )

        print(
            "Automatic ROI boxes:",
            len(boxes),
            flush=True
        )

        # Save ROI coordinates before evaluation.

        with open(
            fold_out / f"{case}_boxes.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                boxes,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Step 4: Frozen fine segmentation
        # ----------------------------------------------------

        (
            prediction,
            union_mask,
            roi_fraction,
            processing_fraction
        ) = segment_rois(
            image,
            boxes,
            fine_model,
            train_mod,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_seconds = (
            time.perf_counter() - t0
        )

        # ----------------------------------------------------
        # Step 5: Save image-only prediction
        # ----------------------------------------------------

        prediction_path = (
            fold_out /
            f"{case}_prediction.nii.gz"
        )

        save_prediction(
            prediction,
            affine,
            factors,
            prediction_path
        )

        # ----------------------------------------------------
        # Step 6: Load GT only after inference
        # ----------------------------------------------------

        _, gt, _, _ = v1.load_volume(
            case,
            need_gt=True
        )

        result = evaluate_prediction(
            prediction,
            union_mask,
            gt
        )

        row = {
            "case_id": case,
            "fold": fold,
            "number_of_boxes": len(boxes),
            "roi_fraction": roi_fraction,
            "processing_fraction": processing_fraction,
            "inference_seconds": inference_seconds,
            **result
        }

        rows.append(row)

        pd.DataFrame(rows).to_csv(
            fold_out / "metrics.csv",
            index=False
        )

        print(
            f"\n{case} RESULT:",
            flush=True
        )

        print(
            f"Dice={result['Dice']:.4f} "
            f"IoU={result['IoU']:.4f} "
            f"Precision={result['Precision']:.4f} "
            f"Recall={result['Recall']:.4f}",
            flush=True
        )

        print(
            f"HD95={result['HD95_mm']:.2f} mm "
            f"ASSD={result['ASSD_mm']:.2f} mm",
            flush=True
        )

        print(
            f"ROI_GT_Coverage="
            f"{result['roi_gt_coverage']:.4f} "
            f"ROI_Fraction={roi_fraction:.4f} "
            f"Processing_Fraction="
            f"{processing_fraction:.4f}",
            flush=True
        )

        print(
            f"Inference={inference_seconds:.2f}s",
            flush=True
        )

        del (
            image,
            probability,
            prediction,
            union_mask,
            gt,
            candidates,
            boxes
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    metrics_columns = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "HD95_mm",
        "ASSD_mm",
        "roi_gt_coverage",
        "roi_fraction",
        "processing_fraction",
        "inference_seconds"
    ]

    summary = df[
        metrics_columns
    ].agg([
        "mean",
        "std",
        "min",
        "max"
    ]).T

    summary.to_csv(
        fold_out / "summary.csv"
    )

    print("\n" + "=" * 100)
    print("FOLD SUMMARY")
    print("=" * 100)

    print(summary.to_string())

    print("\nSaved:", fold_out)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 6)
    )

    args = parser.parse_args()

    evaluate_fold(args.fold)


if __name__ == "__main__":
    main()
